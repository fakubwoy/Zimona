import os, re, json, secrets, time, requests as _requests
from datetime import datetime
from functools import wraps
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import or_, and_, func
from sqlalchemy.orm import joinedload
from werkzeug.utils import secure_filename
from google import genai

# ---------- App & Config ----------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-change-me')
_db_url = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/jewellery')
# Railway (and most managed PG providers) require SSL; add sslmode only when connecting
# to a remote host (i.e. not a local docker-compose db).
_is_local = 'localhost' in _db_url or '@db:' in _db_url

# Build connect_args:
# - On Railway/remote: require SSL + TCP keepalives so the OS detects dead connections
#   before SQLAlchemy attempts to reuse them (fixes "ssl/tls alert bad record mac").
#   keepalives_idle=30  → send first keepalive probe after 30 s of silence
#   keepalives_interval=10 → resend every 10 s while unanswered
#   keepalives_count=5  → declare dead after 5 missed probes (~80 s total)
# - On local docker-compose: no SSL, no keepalives needed.
_connect_args = {} if _is_local else {
    'sslmode': 'require',
    'keepalives': 1,
    'keepalives_idle': 30,
    'keepalives_interval': 10,
    'keepalives_count': 5,
}

app.config['SQLALCHEMY_DATABASE_URI'] = _db_url
# pool_pre_ping: issues a cheap "SELECT 1" before handing out a connection;
#   discards and reconnects if the check fails (catches most stale-connection errors).
# pool_recycle: proactively replace connections older than 60 s — Railway's proxy
#   can silently kill idle SSL sessions, so we recycle well before that happens.
# pool_size / max_overflow kept small; gunicorn sync workers don't need many.
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 60,
    'pool_timeout': 20,
    'pool_size': 5,
    'max_overflow': 2,
    'connect_args': _connect_args,
}
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Use absolute path so uploads work regardless of CWD (important on Railway/Docker)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(_BASE_DIR, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB — client-side pre-compression keeps typical uploads well under this
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}

os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'categories'), exist_ok=True)

db = SQLAlchemy(app)

# Invalidate pooled connections that encounter SSL or operational errors so they
# are never handed back out — forces a fresh connection on the next request.
# Wrapped in app_context so db.engine is accessible at module load time.
from sqlalchemy import event as _sa_event
import psycopg2 as _psycopg2

with app.app_context():
    @_sa_event.listens_for(db.engine, 'handle_error')
    def _handle_db_error(exception_context):
        orig = getattr(exception_context.original_exception, '__cause__', None) \
               or exception_context.original_exception
        if isinstance(orig, (_psycopg2.OperationalError, _psycopg2.InterfaceError)):
            exception_context.invalidate_pool_on_disconnect = True

_genai_client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))

ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'changeme')

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('admin_login', next=request.path))
        return f(*args, **kwargs)
    return decorated

def wholesale_login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('wholesale_buyer_id'):
            return redirect(url_for('wholesale_login', next=request.path))
        buyer = WholesaleBuyer.query.get(session['wholesale_buyer_id'])
        if not buyer or not buyer.is_active:
            session.pop('wholesale_buyer_id', None)
            return redirect(url_for('wholesale_login'))
        return f(*args, **kwargs)
    return decorated

@app.errorhandler(413)
def request_entity_too_large(e):
    from flask import jsonify, request as _req
    if _req.path.startswith('/admin'):
        flash('Image is too large (max 50 MB). Please use a smaller file.', 'warning')
        return redirect(_req.referrer or url_for('admin_dashboard')), 303
    return jsonify({'error': 'File too large. Maximum upload size is 50 MB.'}), 413
_GEMINI_MODEL = 'gemini-2.5-pro'  # Model used for all AI features

# ---------- Context Processor ----------
@app.context_processor
def inject_now():
    return {
        'now': datetime.utcnow(),
        'admin_logged_in': session.get('admin_logged_in', False),
    }

# ---------- Helpers ----------
def slugify(text):
    return re.sub(r'[-\s]+', '-', re.sub(r'[^\w\s-]', '', text.strip().lower()))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def compute_silver_price(product, live_rate_per_kg, premium_per_kg):
    """
    Compute the auto price for a silver-priced product.
    Formula: ((live_rate_per_kg + premium_per_kg) / 1000) * weight_grams * multiplier + fixed_addition
    """
    if not product.silver_pricing_enabled or not product.silver_weight_grams:
        return None
    rate_per_gram = (live_rate_per_kg + premium_per_kg) / 1000.0
    base = rate_per_gram * float(product.silver_weight_grams)
    multiplier = float(product.silver_multiplier) if product.silver_multiplier else 1.0
    fixed = float(product.silver_fixed_addition) if product.silver_fixed_addition else 0.0
    return round(base * multiplier + fixed, 2)

def compress_image(filepath, max_size=(1200, 1200), quality=85):
    with Image.open(filepath) as img:
        img.thumbnail(max_size)
        # Convert to WebP for better compression and Core Web Vitals
        webp_path = os.path.splitext(filepath)[0] + '.webp'
        img.save(webp_path, 'WEBP', optimize=True, quality=quality)
    # Remove original non-webp file if we created a webp version
    if filepath != webp_path and os.path.exists(webp_path):
        os.remove(filepath)
    return webp_path

def save_images(files, name_slug='product'):
    filenames = []
    for f in files:
        if f and allowed_file(f.filename):
            ext = f.filename.rsplit('.', 1)[1].lower()
            # Include slug in filename so Google Image Search gets keyword signals
            # e.g. "solitaire-diamond-ring-a3f8b2c1.webp" instead of "a3f8b2c1.webp"
            filename = f"{name_slug[:40]}-{secrets.token_hex(6)}.{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(filepath)
            webp_path = compress_image(filepath)
            # Store the final filename (always .webp)
            filenames.append(os.path.basename(webp_path))
    return filenames

# ---------- Models ----------
class Category(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    slug = db.Column(db.String(120), unique=True, nullable=False)
    spec_schema = db.Column(db.JSON, default=list)
    image = db.Column(db.String(200), nullable=True)
    description = db.Column(db.Text, nullable=True)
    parent_id = db.Column(db.Integer, db.ForeignKey('category.id'), nullable=True)
    subcategories = db.relationship('Category', backref=db.backref('parent', remote_side='Category.id'), lazy='dynamic')

    @staticmethod
    def seed_defaults():
        pass

    @staticmethod
    def seed_gifts():
        pass
class Settings(db.Model):
    id = db.Column(db.Integer, primary_key=True)          # ← new auto‑increment primary key
    key = db.Column(db.String(100), unique=True, nullable=False)  # ← key is now unique, not PK
    value = db.Column(db.JSON)

    @staticmethod
    def get(key, default=None):
        row = Settings.query.filter_by(key=key).first()   # ← use filter_by, not .get()
        return row.value if row else default

    @staticmethod
    def set(key, value):
        row = Settings.query.filter_by(key=key).first()   # ← use filter_by
        if row:
            row.value = value
        else:
            db.session.add(Settings(key=key, value=value))
        db.session.commit()


class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    slug = db.Column(db.String(220), unique=True, nullable=False)
    price = db.Column(db.Numeric(10,2), nullable=False)
    description = db.Column(db.Text)
    category_id = db.Column(db.Integer, db.ForeignKey('category.id'))
    category = db.relationship('Category', backref='products')
    specs = db.Column(db.JSON)
    images = db.Column(db.JSON, default=list)
    meta_title = db.Column(db.String(60))
    meta_description = db.Column(db.String(160))
    meta_keywords = db.Column(db.String(200))
    tags = db.Column(db.String(500))
    synonyms = db.Column(db.String(500))
    color = db.Column(db.String(200), nullable=True)       # comma-separated color names
    buying_for = db.Column(db.String(200), nullable=True)  # e.g. "Womens,Kids"
    is_in_stock = db.Column(db.Boolean, default=True, nullable=False)
    sort_order = db.Column(db.Integer, default=0, nullable=False)
    rating_value = db.Column(db.Numeric(3, 2), nullable=True)   # e.g. 4.50
    rating_count = db.Column(db.Integer, default=0)              # number of reviews
    created = db.Column(db.DateTime, default=datetime.utcnow)
    updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Silver live pricing
    silver_pricing_enabled = db.Column(db.Boolean, default=False, nullable=False)
    silver_weight_grams    = db.Column(db.Numeric(8, 3), nullable=True)   # weight in grams
    silver_multiplier      = db.Column(db.Numeric(6, 4), nullable=True)   # e.g. 1.15 for 15% making charge
    silver_fixed_addition  = db.Column(db.Numeric(10, 2), nullable=True)  # fixed per-product addition (optional)

    def to_dict(self):
        return {
            'id': self.id, 'name': self.name, 'slug': self.slug,
            'price': float(self.price), 'description': self.description,
            'category': self.category.name if self.category else None,
            'specs': self.specs, 'images': self.images,
            'meta_title': self.meta_title, 'meta_description': self.meta_description,
            'meta_keywords': self.meta_keywords, 'tags': self.tags, 'synonyms': self.synonyms,
            'color': self.color, 'buying_for': self.buying_for,
            'is_in_stock': self.is_in_stock, 'sort_order': self.sort_order,
            'rating_value': float(self.rating_value) if self.rating_value else None,
            'rating_count': self.rating_count,
            'silver_pricing_enabled': self.silver_pricing_enabled,
            'silver_weight_grams': float(self.silver_weight_grams) if self.silver_weight_grams else None,
            'silver_multiplier': float(self.silver_multiplier) if self.silver_multiplier else None,
            'silver_fixed_addition': float(self.silver_fixed_addition) if self.silver_fixed_addition else None,
        }

# ---------- Wholesale Buyer Model ----------
class WholesaleBuyer(db.Model):
    """Invite-only wholesale accounts. Created by admin; login with phone + password."""
    id            = db.Column(db.Integer, primary_key=True)
    phone         = db.Column(db.String(20), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    name          = db.Column(db.String(200), nullable=True)
    company       = db.Column(db.String(200), nullable=True)
    city          = db.Column(db.String(100), nullable=True)
    notes         = db.Column(db.Text, nullable=True)
    is_active     = db.Column(db.Boolean, default=True, nullable=False)
    created       = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, raw):
        import hashlib, os as _os
        salt = _os.urandom(16).hex()
        h = hashlib.sha256((salt + raw).encode()).hexdigest()
        self.password_hash = f"{salt}:{h}"

    def check_password(self, raw):
        import hashlib
        try:
            salt, h = self.password_hash.split(':', 1)
            return hashlib.sha256((salt + raw).encode()).hexdigest() == h
        except Exception:
            return False



# ---------- Routes ----------
@app.route('/')
def index():
    categories = Category.query.filter_by(parent_id=None).all()

    # Top Sellers: use curated IDs if set, else fall back to recent 8
    top_seller_ids = Settings.get('top_seller_ids', [])
    if top_seller_ids:
        top_sellers = Product.query.filter(Product.id.in_(top_seller_ids)).all()
        # preserve order
        id_order = {pid: i for i, pid in enumerate(top_seller_ids)}
        top_sellers = sorted(top_sellers, key=lambda p: id_order.get(p.id, 999))
    else:
        top_sellers = Product.query.order_by(Product.created.desc()).limit(8).all()

    # New Arrivals: use curated IDs if set, else fall back to latest 4
    new_arrival_ids = Settings.get('new_arrival_ids', [])
    if new_arrival_ids:
        new_arrivals = Product.query.filter(Product.id.in_(new_arrival_ids)).all()
        id_order2 = {pid: i for i, pid in enumerate(new_arrival_ids)}
        new_arrivals = sorted(new_arrivals, key=lambda p: id_order2.get(p.id, 999))
    else:
        new_arrivals = Product.query.order_by(Product.created.desc()).limit(4).all()

    featured = top_sellers  # kept for backward compat with template

    brand_promises = Settings.get('brand_promises', [
        {'icon': '925', 'title': 'Fine Silver Jewellery'},
        {'icon': '✦', 'title': '100% Genuine Products'},
        {'icon': '◈', 'title': 'Always Cadmium Free'},
    ])
    budget_ranges = Settings.get('budget_ranges', [
        {'label': 'Gifts Under ₹1499', 'url': '/search?product_type=Gifts&min_price=0&max_price=1499'},
        {'label': 'Gifts ₹1499–₹2499', 'url': '/search?product_type=Gifts&min_price=1499&max_price=2499'},
        {'label': 'Gifts ₹2499–₹4999', 'url': '/search?product_type=Gifts&min_price=2499&max_price=4999'},
        {'label': 'Gifts Above ₹4999', 'url': '/search?product_type=Gifts&min_price=4999'},
    ])
    return render_template('index.html', categories=categories, products=featured,
                           new_arrivals=new_arrivals,
                           brand_promises=brand_promises, budget_ranges=budget_ranges)

@app.route('/product/<slug>')
def product_detail(slug):
    product = Product.query.filter_by(slug=slug).first_or_404()
    return render_template('product.html', product=product)

@app.route('/search')
def search():
    q = request.args.get('q', '').strip()
    min_price = request.args.get('min_price', type=float)
    max_price = request.args.get('max_price', type=float)
    in_stock_only = request.args.get('in_stock') == '1'
    colors_filter = [c.strip() for c in request.args.getlist('color') if c.strip()]
    buying_for_filter = [b.strip() for b in request.args.getlist('buying_for') if b.strip()]
    product_type_filter = [pt.strip() for pt in request.args.getlist('product_type') if pt.strip()]
    sort_by = request.args.get('sort', 'featured')  # featured|price_asc|price_desc|name_asc|name_desc|newest|oldest

    products = []
    page = max(1, request.args.get('page', 1, type=int))
    per_page = 24
    has_filters = bool(q or min_price is not None or max_price is not None
                       or in_stock_only or colors_filter or buying_for_filter or product_type_filter)
    # Always load products — show all when no filters/query
    if True:
        from sqlalchemy.orm import aliased
        ParentCategory = aliased(Category)
        query = Product.query.join(Category, isouter=True).join(
            ParentCategory, Category.parent_id == ParentCategory.id, isouter=True
        )

        if q:
            import re as _re
            pattern = _re.compile(r'(?<![a-zA-Z])' + _re.escape(q) + r'(?![a-zA-Z])', _re.IGNORECASE)
            candidates = query.filter(
                or_(
                    Product.name.ilike(f'%{q}%'),
                    Product.description.ilike(f'%{q}%'),
                    Product.tags.ilike(f'%{q}%'),
                    Product.synonyms.ilike(f'%{q}%'),
                    Product.meta_keywords.ilike(f'%{q}%'),
                    Category.name.ilike(f'%{q}%'),
                    ParentCategory.name.ilike(f'%{q}%'),
                )
            ).all()
            products = [
                p for p in candidates
                if pattern.search(p.name or '')
                or pattern.search(p.description or '')
                or pattern.search(p.tags or '')
                or pattern.search(p.synonyms or '')
                or pattern.search(p.meta_keywords or '')
                or (p.category and pattern.search(p.category.name or ''))
                or (p.category and p.category.parent and pattern.search(p.category.parent.name or ''))
            ]
        else:
            products = query.all()

        # Price filter
        if min_price is not None:
            products = [p for p in products if float(p.price) >= min_price]
        if max_price is not None:
            products = [p for p in products if float(p.price) <= max_price]
        # Stock filter
        if in_stock_only:
            products = [p for p in products if p.is_in_stock]
        # Color filter (any match in comma-separated color field)
        if colors_filter:
            def has_color(p):
                if not p.color:
                    return False
                pcols = [c.strip().lower() for c in p.color.split(',')]
                return any(fc.lower() in pcols for fc in colors_filter)
            products = [p for p in products if has_color(p)]
        # Buying For filter
        if buying_for_filter:
            def has_buying_for(p):
                if not p.buying_for:
                    return False
                pbf = [b.strip().lower() for b in p.buying_for.split(',')]
                return any(f.lower() in pbf for f in buying_for_filter)
            products = [p for p in products if has_buying_for(p)]
        # Product type filter — matches direct category OR subcategories of matched parent
        if product_type_filter:
            matched_cat_ids = set()
            for pt in product_type_filter:
                cat = Category.query.filter(func.lower(Category.name) == pt.lower()).first()
                if cat:
                    matched_cat_ids.add(cat.id)
                    for sub in cat.subcategories.all():
                        matched_cat_ids.add(sub.id)
            if matched_cat_ids:
                products = [p for p in products if p.category_id in matched_cat_ids
                            or (p.category and p.category.parent_id in matched_cat_ids)]
            else:
                products = [p for p in products if p.category and p.category.name.lower() in [pt.lower() for pt in product_type_filter]]

        # Sorting
        if sort_by == 'price_asc':
            products.sort(key=lambda p: float(p.price))
        elif sort_by == 'price_desc':
            products.sort(key=lambda p: float(p.price), reverse=True)
        elif sort_by == 'name_asc':
            products.sort(key=lambda p: p.name.lower())
        elif sort_by == 'name_desc':
            products.sort(key=lambda p: p.name.lower(), reverse=True)
        elif sort_by == 'newest':
            products.sort(key=lambda p: p.created or datetime.min, reverse=True)
        elif sort_by == 'oldest':
            products.sort(key=lambda p: p.created or datetime.min)
        else:  # featured / default
            products.sort(key=lambda p: (-(p.sort_order or 0), p.name.lower()))

    # Pagination
    total_products = len(products)
    total_pages = max(1, (total_products + per_page - 1) // per_page)
    page = min(page, total_pages)
    products_page = products[(page - 1) * per_page : page * per_page]

    # Build facet data for sidebar
    # Use the matched result set for facets so filters only show options that exist
    # in the current results. Fall back to all products only when no query was entered.
    all_products_for_facets = Product.query.join(Category, isouter=True).all()
    facet_source = products if has_filters else all_products_for_facets
    # Buying For facets
    bf_counts = {}
    for p in facet_source:
        for b in (p.buying_for or '').split(','):
            b = b.strip()
            if b:
                bf_counts[b] = bf_counts.get(b, 0) + 1
    # Color facets
    col_counts = {}
    for p in facet_source:
        for c in (p.color or '').split(','):
            c = c.strip()
            if c:
                col_counts[c] = col_counts.get(c, 0) + 1
    # Category (product type) facets — show both parent category AND subcategories
    # e.g. "Gifts (19)", "Silver Coins (3)", "Silver Idols (3)" etc.
    # For top-level categories, count direct products + all subcategory products
    # For subcategories, count their own products only
    cat_counts = {}
    source = facet_source
    for p in source:
        if not p.category:
            continue
        if p.category.parent_id:
            # Product is in a subcategory — add count to subcategory AND to parent
            cat_counts[p.category.name] = cat_counts.get(p.category.name, 0) + 1
            parent_name = p.category.parent.name
            cat_counts[parent_name] = cat_counts.get(parent_name, 0) + 1
        else:
            # Product is directly in a top-level category
            cat_counts[p.category.name] = cat_counts.get(p.category.name, 0) + 1
    # Build cat_tree: ordered list of dicts for the Product Type sidebar
    # Each entry: {name, count, is_sub} — subcategories are indented under their parent
    # We collect parents first, then their subs, preserving a logical order
    _cat_parents = {}   # parent_name -> count
    _cat_subs = {}      # parent_name -> [(sub_name, count)]
    for p in source:
        if not p.category:
            continue
        if p.category.parent_id:
            parent_name = p.category.parent.name
            _cat_parents[parent_name] = _cat_parents.get(parent_name, 0) + 1
            subs = _cat_subs.setdefault(parent_name, {})
            subs[p.category.name] = subs.get(p.category.name, 0) + 1
        else:
            _cat_parents[p.category.name] = _cat_parents.get(p.category.name, 0) + 1

    cat_tree = []
    for parent_name in sorted(_cat_parents):
        cat_tree.append({'name': parent_name, 'count': _cat_parents[parent_name], 'is_sub': False})
        for sub_name in sorted(_cat_subs.get(parent_name, {})):
            cat_tree.append({'name': sub_name, 'count': _cat_subs[parent_name][sub_name], 'is_sub': True})

    # Price range — use facet_source so slider reflects actual results
    all_prices = [float(p.price) for p in (facet_source if facet_source else all_products_for_facets) if p.price]
    global_max_price = int(max(all_prices)) if all_prices else 50000

    return render_template('search.html', query=q, products=products_page,
                           total_products=total_products,
                           page=page, total_pages=total_pages, per_page=per_page,
                           min_price=min_price, max_price=max_price,
                           has_filters=has_filters or sort_by != 'featured',
                           in_stock_only=in_stock_only,
                           colors_filter=colors_filter,
                           buying_for_filter=buying_for_filter,
                           product_type_filter=product_type_filter,
                           sort_by=sort_by,
                           bf_counts=bf_counts,
                           col_counts=col_counts,
                           cat_counts=cat_counts,
                           cat_tree=cat_tree,
                           global_max_price=global_max_price)

@app.route('/robots.txt')
def robots_txt():
    base = request.url_root.rstrip('/')
    content = f"""User-agent: *
Allow: /
Allow: /uploads/
Disallow: /admin
Disallow: /admin/
Disallow: /api/
Disallow: /search?*color=
Disallow: /search?*buying_for=
Disallow: /search?*in_stock=
Disallow: /search?*min_price=
Disallow: /search?*max_price=
Disallow: /search?*sort=

Sitemap: {base}/sitemap.xml
"""
    return app.response_class(content, mimetype='text/plain')

@app.route('/sitemap.xml')
def sitemap_xml():
    from datetime import timezone
    base = request.url_root.rstrip('/')
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    urls = []
    # Homepage
    urls.append({'loc': f'{base}/', 'priority': '1.0', 'changefreq': 'daily', 'lastmod': now})
    # Search / shop-all
    urls.append({'loc': f'{base}/search', 'priority': '0.9', 'changefreq': 'daily', 'lastmod': now})

    # Category pages (proper landing pages, not search params)
    categories = Category.query.filter_by(parent_id=None).all()
    for cat in categories:
        urls.append({
            'loc': f'{base}/category/{cat.slug}',
            'priority': '0.8', 'changefreq': 'weekly', 'lastmod': now
        })
        for sub in cat.subcategories.all():
            urls.append({
                'loc': f'{base}/category/{sub.slug}',
                'priority': '0.7', 'changefreq': 'weekly', 'lastmod': now
            })

    # Product pages — include image tags for Google Image Search
    products = Product.query.all()
    for p in products:
        lastmod = p.updated.strftime('%Y-%m-%d') if p.updated else now
        urls.append({
            'loc': f'{base}/product/{p.slug}',
            'priority': '0.7', 'changefreq': 'monthly', 'lastmod': lastmod,
            'images': [
                {
                    'loc': f'{base}/uploads/{img}',
                    'title': f'{p.name} | Zimona',
                    'caption': (p.meta_description or p.description or p.name or '')[:200]
                }
                for img in (p.images or [])[:5]
            ]
        })

    def _xml_esc(s):
        return (s or '').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    xml_parts = ['<?xml version="1.0" encoding="UTF-8"?>',
                 '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"'
                 ' xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">']
    for u in urls:
        parts = ['  <url>']
        parts.append(f'    <loc>{u["loc"]}</loc>')
        parts.append(f'    <lastmod>{u["lastmod"]}</lastmod>')
        parts.append(f'    <changefreq>{u["changefreq"]}</changefreq>')
        parts.append(f'    <priority>{u["priority"]}</priority>')
        for img in u.get('images', []):
            parts.append('    <image:image>')
            parts.append(f'      <image:loc>{img["loc"]}</image:loc>')
            parts.append(f'      <image:title>{_xml_esc(img["title"])}</image:title>')
            parts.append(f'      <image:caption>{_xml_esc(img["caption"])}</image:caption>')
            parts.append('    </image:image>')
        parts.append('  </url>')
        xml_parts.extend(parts)
    xml_parts.append('</urlset>')
    return app.response_class('\n'.join(xml_parts), mimetype='application/xml')

@app.route('/category/<slug>')
def category_page(slug):
    cat = Category.query.filter_by(slug=slug).first_or_404()
    subcats = cat.subcategories.all()
    parent = cat.parent if cat.parent_id else None

    # Collect all category IDs in scope (this cat + its subcats)
    cat_ids = {cat.id} | {s.id for s in subcats}

    # Base product set for this category
    all_cat_products = Product.query.filter(Product.category_id.in_(cat_ids)).all()

    # --- Filters ---
    min_price = request.args.get('min_price', type=float)
    max_price = request.args.get('max_price', type=float)
    in_stock_only = request.args.get('in_stock') == '1'
    colors_filter = [c.strip() for c in request.args.getlist('color') if c.strip()]
    buying_for_filter = [b.strip() for b in request.args.getlist('buying_for') if b.strip()]
    sort_by = request.args.get('sort', 'featured')
    active_subcat = request.args.get('subcat', '').strip()  # slug of selected subcategory

    products = list(all_cat_products)
    # Filter by subcategory if selected
    if active_subcat:
        sub_match = next((s for s in subcats if s.slug == active_subcat), None)
        if sub_match:
            products = [p for p in products if p.category_id == sub_match.id]
    if min_price is not None:
        products = [p for p in products if float(p.price) >= min_price]
    if max_price is not None:
        products = [p for p in products if float(p.price) <= max_price]
    if in_stock_only:
        products = [p for p in products if p.is_in_stock]
    if colors_filter:
        products = [p for p in products if p.color and any(
            c.lower() in [x.strip().lower() for x in p.color.split(',')] for c in colors_filter)]
    if buying_for_filter:
        products = [p for p in products if p.buying_for and any(
            b.lower() in [x.strip().lower() for x in p.buying_for.split(',')] for b in buying_for_filter)]

    # --- Sort ---
    if sort_by == 'price_asc':
        products.sort(key=lambda p: float(p.price))
    elif sort_by == 'price_desc':
        products.sort(key=lambda p: float(p.price), reverse=True)
    elif sort_by == 'name_asc':
        products.sort(key=lambda p: p.name.lower())
    elif sort_by == 'name_desc':
        products.sort(key=lambda p: p.name.lower(), reverse=True)
    elif sort_by == 'newest':
        products.sort(key=lambda p: p.created or datetime.min, reverse=True)
    elif sort_by == 'oldest':
        products.sort(key=lambda p: p.created or datetime.min)
    else:
        products.sort(key=lambda p: (-(p.sort_order or 0), p.name.lower()))

    # --- Facets (computed from unfiltered base set) ---
    bf_counts = {}
    col_counts = {}
    for p in all_cat_products:
        for b in (p.buying_for or '').split(','):
            b = b.strip()
            if b: bf_counts[b] = bf_counts.get(b, 0) + 1
        for c in (p.color or '').split(','):
            c = c.strip()
            if c: col_counts[c] = col_counts.get(c, 0) + 1

    all_prices = [float(p.price) for p in all_cat_products if p.price]
    global_max_price = int(max(all_prices)) if all_prices else 50000

    has_filters = bool(min_price is not None or max_price is not None or
                       in_stock_only or colors_filter or buying_for_filter)

    return render_template('category.html', category=cat, products=products,
                           subcats=subcats, parent=parent,
                           sort_by=sort_by, has_filters=has_filters,
                           min_price=min_price, max_price=max_price,
                           in_stock_only=in_stock_only,
                           colors_filter=colors_filter,
                           buying_for_filter=buying_for_filter,
                           bf_counts=bf_counts, col_counts=col_counts,
                           global_max_price=global_max_price,
                           active_subcat=active_subcat,
                           total_products=len(products))

@app.route('/feed.xml')
def merchant_feed():
    """Google Merchant Center / Shopping product feed (RSS2 format)."""
    base = request.url_root.rstrip('/')
    products = Product.query.filter_by(is_in_stock=True).all()
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<rss version="2.0" xmlns:g="http://base.google.com/ns/1.0">',
             '<channel>',
             f'<title>Zimona Silver Jewellery</title>',
             f'<link>{base}/</link>',
             f'<description>925 Sterling Silver Jewellery – Zimona</description>']
    for p in products:
        img_url = f'{base}/uploads/{p.images[0]}' if p.images else ''
        cat_name = p.category.name if p.category else 'Jewellery'
        desc = (p.description or p.name).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        name = p.name.replace('&', '&amp;')
        lines.append('<item>')
        lines.append(f'<g:id>{p.id}</g:id>')
        lines.append(f'<g:title>{name}</g:title>')
        lines.append(f'<g:description>{desc}</g:description>')
        lines.append(f'<g:link>{base}/product/{p.slug}</g:link>')
        if img_url:
            lines.append(f'<g:image_link>{img_url}</g:image_link>')
        lines.append(f'<g:price>{float(p.price):.2f} INR</g:price>')
        lines.append(f'<g:availability>in stock</g:availability>')
        lines.append(f'<g:condition>new</g:condition>')
        lines.append(f'<g:brand>Zimona</g:brand>')
        lines.append(f'<g:google_product_category>Apparel &amp; Accessories &gt; Jewelry</g:google_product_category>')
        lines.append(f'<g:product_type>{cat_name}</g:product_type>')
        material = (p.specs or {}).get('Material', '925 Sterling Silver')
        lines.append(f'<g:material>{material}</g:material>')
        if p.color:
            lines.append(f'<g:color>{p.color.split(",")[0].strip()}</g:color>')
        lines.append('</item>')
    lines.extend(['</channel>', '</rss>'])
    return app.response_class('\n'.join(lines), mimetype='application/xml')

@app.route('/api/search')
def api_search():
    q = request.args.get('q', '').strip()
    if not q or len(q) < 2:
        return jsonify([])
    import re as _re
    pattern = _re.compile(r'(?<![a-zA-Z])' + _re.escape(q) + r'(?![a-zA-Z])', _re.IGNORECASE)
    candidates = Product.query.join(Category, isouter=True).filter(
        or_(
            Product.name.ilike(f'%{q}%'),
            Product.tags.ilike(f'%{q}%'),
            Product.synonyms.ilike(f'%{q}%'),
            Product.meta_keywords.ilike(f'%{q}%'),
            Category.name.ilike(f'%{q}%'),
        )
    ).limit(20).all()
    results = [
        p for p in candidates
        if pattern.search(p.name or '')
        or pattern.search(p.tags or '')
        or pattern.search(p.synonyms or '')
        or pattern.search(p.meta_keywords or '')
        or (p.category and pattern.search(p.category.name or ''))
    ]
    return jsonify([{
        'id': p.id,
        'name': p.name,
        'slug': p.slug,
        'price': float(p.price),
        'category': p.category.name if p.category else '',
        'thumb': ('/uploads/' + p.images[0]) if p.images else '',
        'tags': (p.tags or '925 Silver').split(',')[0].strip(),
    } for p in results[:12]])

# ---------- Admin Auth ----------
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if session.get('admin_logged_in'):
        return redirect(url_for('admin_dashboard'))
    error = None
    if request.method == 'POST':
        if (request.form.get('username') == ADMIN_USERNAME and
                request.form.get('password') == ADMIN_PASSWORD):
            session['admin_logged_in'] = True
            session.permanent = False
            next_url = request.args.get('next') or url_for('admin_dashboard')
            return redirect(next_url)
        error = 'Invalid username or password.'
    return render_template('admin/login.html', error=error)

@app.route('/admin/logout', methods=['POST'])
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))

# ---------- Admin ----------
@app.route('/admin')
@login_required
def admin_dashboard():
    # Pass all categories; template will group by parent_id
    categories = Category.query.options(joinedload(Category.products)).filter_by(parent_id=None).all()
    # Products with no category assigned — always accessible regardless of category state
    uncategorised_products = Product.query.filter_by(category_id=None).order_by(Product.name).all()
    silver_settings = {
        'live_rate_per_kg': Settings.get('silver_live_rate_per_kg', 0) or 0,
        'premium_per_kg':   Settings.get('silver_premium_per_kg',   0) or 0,
        'auto_update':      Settings.get('silver_auto_update',       False),
        'updated_at':       Settings.get('silver_rate_updated_at',   None),
    }
    return render_template('admin/dashboard.html', categories=categories,
                           uncategorised_products=uncategorised_products,
                           silver_settings=silver_settings)

@app.route('/admin/product/new', methods=['GET', 'POST'])
@app.route('/admin/product/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def admin_product_form(id=None):
    product = Product.query.get(id) if id else None
    # Build ordered list: top-level cats first, then their subcategories indented
    top_level = Category.query.filter_by(parent_id=None).order_by(Category.name).all()
    categories = []
    for cat in top_level:
        categories.append(cat)
        for sub in cat.subcategories.order_by(Category.name).all():
            sub._display_name = f'\u00a0\u00a0\u00a0\u00a0↳ {sub.name}'
            categories.append(sub)
    # ensure plain cats without parent also listed (already top_level)
    if request.method == 'POST':
        name = request.form['name']
        price = request.form['price']
        description = request.form['description']
        category_id = request.form.get('category_id')
        specs_json = request.form.get('specs', '{}')
        try:
            specs = json.loads(specs_json) if specs_json else {}
        except:
            specs = {}
        meta_title = request.form.get('meta_title', '')
        meta_description = request.form.get('meta_description', '')
        meta_keywords = request.form.get('meta_keywords', '')
        tags = request.form.get('tags', '')
        synonyms = request.form.get('synonyms', '')
        color = request.form.get('color', '')
        buying_for = request.form.get('buying_for', '')
        is_in_stock = request.form.get('is_in_stock') == '1'
        sort_order = int(request.form.get('sort_order', 0) or 0)
        rating_value_raw = request.form.get('rating_value', '').strip()
        rating_count_raw = request.form.get('rating_count', '').strip()
        rating_value = float(rating_value_raw) if rating_value_raw else None
        rating_count = int(rating_count_raw) if rating_count_raw else 0

        # Silver live pricing fields
        silver_pricing_enabled = request.form.get('silver_pricing_enabled') == '1'
        silver_weight_raw = request.form.get('silver_weight_grams', '').strip()
        silver_mult_raw   = request.form.get('silver_multiplier', '').strip()
        silver_fixed_raw  = request.form.get('silver_fixed_addition', '').strip()
        silver_weight_grams   = float(silver_weight_raw) if silver_weight_raw else None
        silver_multiplier     = float(silver_mult_raw)   if silver_mult_raw   else None
        silver_fixed_addition = float(silver_fixed_raw)  if silver_fixed_raw  else None

        if not product:
            product = Product()
            db.session.add(product)
        product.name = name
        product.slug = slugify(name)
        product.price = price
        product.description = description
        product.category_id = int(category_id) if category_id else None
        product.specs = specs
        product.meta_title = meta_title[:60]
        product.meta_description = meta_description[:160]
        product.meta_keywords = meta_keywords[:200]
        product.tags = tags[:500]
        product.synonyms = synonyms[:500]
        product.color = color[:200] if color else None
        product.buying_for = buying_for[:200] if buying_for else None
        product.is_in_stock = is_in_stock
        product.sort_order = sort_order
        product.rating_value = rating_value
        product.rating_count = rating_count
        product.silver_pricing_enabled = silver_pricing_enabled
        product.silver_weight_grams    = silver_weight_grams
        product.silver_multiplier      = silver_multiplier
        product.silver_fixed_addition  = silver_fixed_addition

        # If silver pricing is enabled, auto-compute price from current live rate
        if silver_pricing_enabled and silver_weight_grams:
            live_rate = Settings.get('silver_live_rate_per_kg', 0) or 0
            premium   = Settings.get('silver_premium_per_kg', 0) or 0
            computed  = compute_silver_price(product, float(live_rate), float(premium))
            if computed is not None:
                product.price = computed

        # Fix: Always explicitly update images array with exactly what the form sent. 
        # This resolves an issue where deleting all images wouldn't reflect on the server.
        new_images = save_images(request.files.getlist('images'), name_slug=slugify(name)) if 'images' in request.files else []
        kept_images = request.form.getlist('images_keep')
        product.images = kept_images + new_images

        db.session.commit()
        flash('Product saved.', 'success')
        return redirect(url_for('admin_dashboard'))
    return render_template('admin/product_form.html', product=product, categories=categories)

@app.route('/admin/product/<int:id>/delete', methods=['POST'])
@login_required
def admin_product_delete(id):
    product = Product.query.get_or_404(id)
    db.session.delete(product)
    db.session.commit()
    flash('Product deleted.', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/featured', methods=['GET', 'POST'])
@login_required
def admin_featured():
    all_products = Product.query.order_by(Product.name).all()
    if request.method == 'POST':
        section = request.form.get('section')  # 'top_sellers' or 'new_arrivals'
        ids = request.form.getlist('product_ids')
        ids = [int(i) for i in ids if i.isdigit()]
        if section == 'top_sellers':
            Settings.set('top_seller_ids', ids)
        elif section == 'new_arrivals':
            Settings.set('new_arrival_ids', ids)
        flash(f'{"Top Sellers" if section == "top_sellers" else "New Arrivals"} updated.', 'success')
        return redirect(url_for('admin_featured'))
    top_seller_ids = Settings.get('top_seller_ids', [])
    new_arrival_ids = Settings.get('new_arrival_ids', [])
    return render_template('admin/featured.html',
                           all_products=all_products,
                           top_seller_ids=top_seller_ids,
                           new_arrival_ids=new_arrival_ids)

@app.route('/admin/featured/reorder', methods=['POST'])
@login_required
def admin_featured_reorder():
    data = request.json
    section = data.get('section')
    ids = [int(i) for i in data.get('ids', []) if str(i).isdigit()]
    if section == 'top_sellers':
        Settings.set('top_seller_ids', ids)
    elif section == 'new_arrivals':
        Settings.set('new_arrival_ids', ids)
    return jsonify({'ok': True, 'count': len(ids)})

@app.route('/admin/homepage', methods=['GET', 'POST'])
@login_required
def admin_homepage():
    if request.method == 'POST':
        # Budget ranges
        labels = request.form.getlist('budget_label')
        urls = request.form.getlist('budget_url')
        budget_ranges = [{'label': l, 'url': u} for l, u in zip(labels, urls) if l.strip()]
        Settings.set('budget_ranges', budget_ranges)
        # Brand promises
        icons = request.form.getlist('promise_icon')
        titles = request.form.getlist('promise_title')
        brand_promises = [{'icon': ic, 'title': t} for ic, t in zip(icons, titles) if t.strip()]
        Settings.set('brand_promises', brand_promises)
        flash('Homepage settings saved.', 'success')
        return redirect(url_for('admin_homepage'))
    budget_ranges = Settings.get('budget_ranges', [
        {'label': 'Gifts Under ₹1499', 'url': '/search?product_type=Gifts&min_price=0&max_price=1499'},
        {'label': 'Gifts ₹1499–₹2499', 'url': '/search?product_type=Gifts&min_price=1499&max_price=2499'},
        {'label': 'Gifts ₹2499–₹4999', 'url': '/search?product_type=Gifts&min_price=2499&max_price=4999'},
        {'label': 'Gifts Above ₹4999', 'url': '/search?product_type=Gifts&min_price=4999'},
    ])
    brand_promises = Settings.get('brand_promises', [
        {'icon': '925', 'title': 'Fine Silver Jewellery'},
        {'icon': '✦', 'title': '100% Genuine Products'},
        {'icon': '◈', 'title': 'Always Cadmium Free'},
    ])
    return render_template('admin/homepage.html', budget_ranges=budget_ranges, brand_promises=brand_promises)


@app.route('/admin/categories', methods=['GET', 'POST'])
@login_required
def admin_categories():
    if request.method == 'POST':
        name = request.form['name']
        parent_id = request.form.get('parent_id', type=int)
        slug = slugify(name)
        if not Category.query.filter_by(slug=slug).first():
            db.session.add(Category(name=name, slug=slug, spec_schema=[], parent_id=parent_id))
            db.session.commit()
            flash('Category added.', 'success')
        else:
            flash('Category already exists.', 'warning')
        return redirect(url_for('admin_categories'))
    categories = Category.query.filter_by(parent_id=None).all()
    return render_template('admin/categories.html', categories=categories)

@app.route('/admin/category/<int:id>/delete', methods=['POST'])
@login_required
def admin_category_delete(id):
    cat = Category.query.get_or_404(id)
    db.session.delete(cat)
    db.session.commit()
    flash('Category deleted.', 'success')
    return redirect(url_for('admin_categories'))
@app.route('/admin/category/<int:id>/edit', methods=['POST'])
@login_required
def admin_category_edit(id):
    cat = Category.query.get_or_404(id)
    new_name = request.form.get('name', '').strip()
    if not new_name:
        flash('Category name cannot be empty.', 'warning')
        return redirect(url_for('admin_categories'))
    # check for duplicate name (case‑insensitive)
    existing = Category.query.filter(func.lower(Category.name) == new_name.lower()).first()
    if existing and existing.id != id:
        flash('A category with that name already exists.', 'warning')
    else:
        cat.name = new_name
        cat.slug = slugify(new_name)
        cat.description = request.form.get('description', '').strip() or None
        db.session.commit()
        flash('Category saved.', 'success')
    return redirect(url_for('admin_categories'))

@app.route('/admin/category/<int:id>/upload-image', methods=['POST'])
@login_required
def admin_category_upload_image(id):
    cat = Category.query.get_or_404(id)
    f = request.files.get('image')
    if f and allowed_file(f.filename):
        ext = f.filename.rsplit('.', 1)[1].lower()
        filename = f"cat_{secrets.token_hex(6)}.{ext}"

        # Save into static/uploads/categories — this folder lives inside the
        # Docker volume so images survive container redeploys.
        cat_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'categories')
        os.makedirs(cat_upload_dir, exist_ok=True)

        filepath = os.path.join(cat_upload_dir, filename)
        f.save(filepath)
        webp_path = compress_image(filepath)
        # Store just the bare filename (no path prefix).
        # The route /assets/categories/<filename> serves from this same folder.
        cat.image = os.path.basename(webp_path)
        db.session.commit()
        flash('Category image updated.', 'success')
    else:
        flash('Invalid file.', 'warning')
    return redirect(url_for('admin_categories'))

# ---------- AJAX Endpoints for Category & AI Specs ----------
@app.route('/admin/category/add-ajax', methods=['POST'])
@login_required
def add_category_ajax():
    name = request.json.get('name', '').strip()
    parent_id = request.json.get('parent_id', None)
    if not name:
        return jsonify({'error': 'Name required'}), 400
    slug = slugify(name)
    if Category.query.filter_by(slug=slug).first():
        return jsonify({'error': 'Category already exists'}), 400
    cat = Category(name=name, slug=slug, spec_schema=[], parent_id=parent_id)
    db.session.add(cat)
    db.session.commit()
    return jsonify({'id': cat.id, 'name': cat.name})

@app.route('/admin/ai-suggest-specs', methods=['POST'])
@login_required
def ai_suggest_specs():
    data = request.json
    name = data.get('name', '')
    desc = data.get('desc', '')
    category = data.get('category', '')
    prompt = f"""Suggest jewellery specifications for: name="{name}", description="{desc}", category="{category}". Output only JSON object with 4-6 key-value pairs (e.g., {{"Material": "18K Gold", "Weight": "5g"}}). Use common jewellery attributes. No explanations."""
    try:
        resp = model.generate_content(prompt, generation_config={"temperature": 0.1})
        text = resp.text.strip().replace('```json','').replace('```','')
        specs = json.loads(text)
        return jsonify(specs)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ---------- API for Category Spec Keys from Existing Products ----------
@app.route('/api/category/<int:id>/spec-keys')
def category_spec_keys(id):
    cat = Category.query.get_or_404(id)
    products = Product.query.filter_by(category_id=id).all()
    keys = set()
    for p in products:
        if p.specs:
            keys.update(p.specs.keys())
    return jsonify(sorted(list(keys)))

# ---------- API: Get distinct values for a spec key in a category ----------
@app.route('/api/category/<int:id>/spec-values')
def category_spec_values(id):
    key = request.args.get('key', '')
    if not key:
        return jsonify([])
    products = Product.query.filter_by(category_id=id).all()
    values = set()
    for p in products:
        if p.specs and key in p.specs:
            val = p.specs[key]
            if val:
                values.add(str(val))
    return jsonify(sorted(list(values)))

# ---------- Silver Rate Settings ----------
@app.route('/admin/silver-rate', methods=['GET', 'POST'])
@login_required
def admin_silver_rate():
    if request.method == 'POST':
        data = request.json or {}
        live_rate = data.get('live_rate_per_kg')
        premium   = data.get('premium_per_kg', 0)
        auto_update = data.get('auto_update', False)
        if live_rate is None:
            return jsonify({'error': 'live_rate_per_kg required'}), 400
        Settings.set('silver_live_rate_per_kg', float(live_rate))
        Settings.set('silver_premium_per_kg',   float(premium))
        Settings.set('silver_auto_update',       bool(auto_update))
        Settings.set('silver_rate_updated_at',   datetime.utcnow().isoformat())
        updated = 0
        if auto_update:
            products = Product.query.filter_by(silver_pricing_enabled=True).all()
            for p in products:
                new_price = compute_silver_price(p, float(live_rate), float(premium))
                if new_price is not None:
                    p.price = new_price
                    updated += 1
            db.session.commit()
        return jsonify({
            'ok': True,
            'live_rate_per_kg': float(live_rate),
            'premium_per_kg': float(premium),
            'updated_products': updated,
            'updated_at': Settings.get('silver_rate_updated_at'),
        })
    # GET – return current settings
    return jsonify({
        'live_rate_per_kg': Settings.get('silver_live_rate_per_kg', 0),
        'premium_per_kg':   Settings.get('silver_premium_per_kg',   0),
        'auto_update':      Settings.get('silver_auto_update',       False),
        'updated_at':       Settings.get('silver_rate_updated_at',   None),
    })


@app.route('/admin/silver-rate/preview', methods=['POST'])
@login_required
def silver_rate_preview():
    """Return updated prices for all silver-priced products without saving."""
    data = request.json or {}
    live_rate = float(data.get('live_rate_per_kg', 0))
    premium   = float(data.get('premium_per_kg', 0))
    products  = Product.query.filter_by(silver_pricing_enabled=True).all()
    preview   = []
    for p in products:
        new_price = compute_silver_price(p, live_rate, premium)
        if new_price is not None:
            preview.append({
                'id': p.id,
                'name': p.name,
                'current_price': float(p.price),
                'new_price': new_price,
                'weight_grams': float(p.silver_weight_grams),
                'multiplier': float(p.silver_multiplier) if p.silver_multiplier else 1.0,
                'category': p.category.name if p.category else '—',
            })
    return jsonify({'products': preview, 'count': len(preview)})

@app.route('/admin/silver-rate/fetch-live', methods=['GET'])
@login_required
def fetch_live_silver_rate():
    """
    Fetch today's live silver spot price in INR per kg from free public APIs.
    Tries multiple sources in order; returns the first successful result.
    """
    # --- Source 1: metals-api.com (free tier, 50 req/month) ---
    # Requires METALS_API_KEY env var. Falls through gracefully if not set.
    metals_api_key = os.environ.get('METALS_API_KEY', '')
    if metals_api_key:
        try:
            r = _requests.get(
                'https://metals-api.com/api/latest',
                params={'access_key': metals_api_key, 'base': 'INR', 'symbols': 'XAG'},
                timeout=6
            )
            d = r.json()
            if d.get('success') and d.get('rates', {}).get('XAG'):
                # XAG rate = how many XAG per 1 INR  →  invert to get INR per troy oz
                inr_per_troy_oz = 1.0 / float(d['rates']['XAG'])
                inr_per_kg = round(inr_per_troy_oz * 32.1507, 2)   # 1 kg = 32.1507 troy oz
                return jsonify({
                    'rate_per_kg': inr_per_kg,
                    'source': 'metals-api.com',
                    'timestamp': datetime.utcnow().strftime('%d %b %Y, %H:%M UTC'),
                })
        except Exception:
            pass

    # --- Source 2: metalpriceapi.com (free tier, 100 req/month) ---
    metal_price_api_key = os.environ.get('METAL_PRICE_API_KEY', '')
    if metal_price_api_key:
        try:
            r = _requests.get(
                'https://api.metalpriceapi.com/v1/latest',
                params={'api_key': metal_price_api_key, 'base': 'INR', 'currencies': 'XAG'},
                timeout=6
            )
            d = r.json()
            if d.get('success') and d.get('rates', {}).get('XAG'):
                inr_per_troy_oz = 1.0 / float(d['rates']['XAG'])
                inr_per_kg = round(inr_per_troy_oz * 32.1507, 2)
                return jsonify({
                    'rate_per_kg': inr_per_kg,
                    'source': 'metalpriceapi.com',
                    'timestamp': datetime.utcnow().strftime('%d %b %Y, %H:%M UTC'),
                })
        except Exception:
            pass

    # --- Source 3: Frankfurter / exchangerate + manual silver USD→INR conversion ---
    # Uses exchangerate-api.com (free, no key needed) for USD→INR,
    # and the open metals endpoint from commodity-price-api for silver in USD.
    try:
        # Step A: get silver price in USD per troy oz (open, no-key)
        silver_r = _requests.get(
            'https://query1.finance.yahoo.com/v8/finance/chart/SI%3DF',
            headers={'User-Agent': 'Mozilla/5.0'},
            params={'interval': '1d', 'range': '1d'},
            timeout=6
        )
        silver_data = silver_r.json()
        silver_usd_per_oz = float(
            silver_data['chart']['result'][0]['meta']['regularMarketPrice']
        )

        # Step B: get USD → INR rate (open, no key needed)
        fx_r = _requests.get(
            'https://api.frankfurter.app/latest',
            params={'from': 'USD', 'to': 'INR'},
            timeout=5
        )
        fx_data = fx_r.json()
        usd_to_inr = float(fx_data['rates']['INR'])

        inr_per_troy_oz = silver_usd_per_oz * usd_to_inr
        inr_per_kg = round(inr_per_troy_oz * 32.1507, 2)
        return jsonify({
            'rate_per_kg': inr_per_kg,
            'source': 'Yahoo Finance + Frankfurter FX',
            'timestamp': datetime.utcnow().strftime('%d %b %Y, %H:%M UTC'),
        })
    except Exception as e:
        pass

    # --- All sources failed ---
    return jsonify({
        'error': (
            'Could not fetch live silver rate automatically. '
            'Set METALS_API_KEY or METAL_PRICE_API_KEY environment variables for '
            'a more reliable feed (metals-api.com or metalpriceapi.com offer free tiers). '
            'You can also enter the rate manually.'
        )
    }), 503



@app.route('/admin/ai-seo', methods=['POST'])
@login_required
def ai_seo():
    data = request.json
    name = data.get('name', '')
    desc = data.get('desc', '')
    prompt = f"""Generate SEO for jewellery: name="{name}", desc="{desc}". Output JSON with keys: meta_title(max60), meta_description(max160), meta_keywords(comma), tags(comma), synonyms(comma). Only JSON."""
    try:
        resp = _genai_client.models.generate_content(model=_GEMINI_MODEL, contents=prompt, config={"temperature":0.2})
        text = resp.text.strip().replace('```json','').replace('```','')
        seo = json.loads(text)
        return jsonify(seo)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/ai-assistant', methods=['POST'])
@login_required
def ai_assistant():
    data = request.json
    nl = data.get('command', '')
    prompt = f"""Convert this command to a JSON action for jewellery admin: "{nl}".
Output only JSON with keys:
- action: "update_price", "delete_product", etc.
- filters: object with any of: "category" (string), "product_name" (string, partial match), "product_slug" (string).
- value: new value (for update actions).
- value_multiplier: number (for percentage increases, e.g., 1.05 for 5% increase).

Examples:
"make the price of solitaire diamond ring 130000" → {{"action":"update_price","filters":{{"product_name":"solitaire diamond ring"}},"value":130000}}
"increase all ring prices by 5%" → {{"action":"update_price","filters":{{"category":"rings"}},"value_multiplier":1.05}}

Only JSON, no explanation."""
    try:
        resp = _genai_client.models.generate_content(model=_GEMINI_MODEL, contents=prompt, config={"temperature":0.0})
        text = resp.text.strip().replace('```json','').replace('```','')
        action = json.loads(text)
        return jsonify(action)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/apply-action', methods=['POST'])
@login_required
def apply_action():
    action = request.json
    if not action:
        return jsonify({'error': 'No action provided'}), 400

    action_type = action.get('action')
    filters = action.get('filters', {})
    value = action.get('value')
    multiplier = action.get('value_multiplier')

    if action_type == 'update_price':
        query = Product.query

        if 'category' in filters:
            cat = Category.query.filter(func.lower(Category.name) == filters['category'].lower()).first()
            if cat:
                query = query.filter(Product.category_id == cat.id)
            else:
                return jsonify({'error': f"Category '{filters['category']}' not found"}), 404

        if 'product_name' in filters:
            query = query.filter(Product.name.ilike(f"%{filters['product_name']}%"))

        if 'product_slug' in filters:
            query = query.filter(Product.slug.ilike(f"%{filters['product_slug']}%"))

        count = query.count()
        if count == 0:
            return jsonify({'error': 'No matching products found'}), 404

        if value is not None:
            query.update({Product.price: value}, synchronize_session=False)
        elif multiplier is not None:
            for p in query.all():
                p.price = float(p.price) * multiplier
        else:
            return jsonify({'error': 'Missing value or multiplier'}), 400

        db.session.commit()
        return jsonify({'updated': count, 'filters_applied': filters})

    return jsonify({'error': 'Unsupported action'}), 400

@app.route('/api/products')
def api_products():
    products = Product.query.all()
    return jsonify([p.to_dict() for p in products])

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/assets/categories/<path:filename>')
def category_assets(filename):
    # FIX: Intercept category image requests and serve them from the Docker Volume
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], 'categories'), filename)

@app.route('/assets/<path:filename>')
def assets_file(filename):
    return send_from_directory(os.path.join(_BASE_DIR, 'templates', 'assets'), filename)

# ═══════════════════════════════════════════════════════════════════════
#  WHOLESALE PORTAL ROUTES
# ═══════════════════════════════════════════════════════════════════════

@app.route('/wholesale/login', methods=['GET', 'POST'])
def wholesale_login():
    if session.get('wholesale_buyer_id'):
        return redirect(url_for('wholesale_catalog'))
    error = None
    if request.method == 'POST':
        phone = request.form.get('phone', '').strip()
        password = request.form.get('password', '')
        buyer = WholesaleBuyer.query.filter_by(phone=phone, is_active=True).first()
        if buyer and buyer.check_password(password):
            session['wholesale_buyer_id'] = buyer.id
            return redirect(request.args.get('next') or url_for('wholesale_catalog'))
        error = 'Invalid phone number or password.'
    return render_template('wholesale/login.html', error=error)

@app.route('/wholesale/logout')
def wholesale_logout():
    session.pop('wholesale_buyer_id', None)
    return redirect(url_for('wholesale_login'))

@app.route('/wholesale')
@app.route('/wholesale/catalog')
@wholesale_login_required
def wholesale_catalog():
    buyer = WholesaleBuyer.query.get(session['wholesale_buyer_id'])
    q             = request.args.get('q', '').strip()
    cat_id        = request.args.get('cat', type=int)
    sort_by       = request.args.get('sort', 'featured')
    in_stock_only = request.args.get('in_stock') == '1'

    query = Product.query.join(Category, isouter=True)
    if q:
        query = query.filter(or_(
            Product.name.ilike(f'%{q}%'),
            Product.tags.ilike(f'%{q}%'),
            Product.description.ilike(f'%{q}%'),
            Category.name.ilike(f'%{q}%'),
        ))
    if cat_id:
        cat = Category.query.get(cat_id)
        if cat:
            sub_ids = [s.id for s in cat.subcategories.all()]
            all_ids = [cat_id] + sub_ids
            query = query.filter(Product.category_id.in_(all_ids))
    if in_stock_only:
        query = query.filter(Product.is_in_stock == True)

    products = query.all()

    if sort_by == 'price_asc':
        products.sort(key=lambda p: float(p.price))
    elif sort_by == 'price_desc':
        products.sort(key=lambda p: float(p.price), reverse=True)
    elif sort_by == 'weight_asc':
        products.sort(key=lambda p: float(p.silver_weight_grams or 0))
    elif sort_by == 'weight_desc':
        products.sort(key=lambda p: float(p.silver_weight_grams or 0), reverse=True)
    elif sort_by == 'name_asc':
        products.sort(key=lambda p: p.name.lower())
    else:
        products.sort(key=lambda p: (-(p.sort_order or 0), p.name.lower()))

    categories = Category.query.filter_by(parent_id=None).order_by(Category.name).all()
    active_cat = Category.query.get(cat_id) if cat_id else None

    # Group products by top-level category for the default view
    # When filtering by cat or searching, skip grouping (show flat list)
    grouped_products = []
    if not cat_id and not q:
        from collections import OrderedDict
        cat_groups = OrderedDict()
        uncategorised = []
        for p in products:
            if p.category:
                top = p.category.parent if p.category.parent_id else p.category
                key = top.id
                if key not in cat_groups:
                    cat_groups[key] = {
                        'cat_name': top.name,
                        'cat_slug': top.slug,
                        'cat_id': top.id,
                        'products': []
                    }
                cat_groups[key]['products'].append(p)
            else:
                uncategorised.append(p)
        grouped_products = list(cat_groups.values())
        if uncategorised:
            grouped_products.append({
                'cat_name': 'Uncategorised',
                'cat_slug': '',
                'cat_id': None,
                'products': uncategorised
            })

    live_rate = float(Settings.get('silver_live_rate_per_kg', 0) or 0)
    premium   = float(Settings.get('silver_premium_per_kg',   0) or 0)
    eff_rate_per_gram = (live_rate + premium) / 1000.0

    return render_template('wholesale/catalog.html',
        buyer=buyer, products=products, categories=categories,
        grouped_products=grouped_products,
        active_cat=active_cat, q=q, sort_by=sort_by,
        in_stock_only=in_stock_only, total=len(products),
        eff_rate_per_gram=eff_rate_per_gram)

@app.route('/wholesale/product/<slug>')
@wholesale_login_required
def wholesale_product(slug):
    buyer   = WholesaleBuyer.query.get(session['wholesale_buyer_id'])
    product = Product.query.filter_by(slug=slug).first_or_404()

    # Pass live silver rate so template can compute silver multiple
    live_rate = float(Settings.get('silver_live_rate_per_kg', 0) or 0)
    premium   = float(Settings.get('silver_premium_per_kg',   0) or 0)
    effective_rate_per_gram = (live_rate + premium) / 1000.0  # ₹ per gram

    # Silver multiple = product price / (silver cost of the item)
    silver_multiple = None
    if effective_rate_per_gram > 0 and product.silver_weight_grams:
        silver_cost = effective_rate_per_gram * float(product.silver_weight_grams)
        if silver_cost > 0:
            silver_multiple = round(float(product.price) / silver_cost, 2)

    return render_template('wholesale/product.html',
        buyer=buyer, product=product,
        live_rate=live_rate, premium=premium,
        effective_rate_per_gram=effective_rate_per_gram,
        silver_multiple=silver_multiple)

@app.route('/admin/wholesale/count')
@login_required
def admin_wholesale_count():
    active = WholesaleBuyer.query.filter_by(is_active=True).count()
    total  = WholesaleBuyer.query.count()
    return jsonify({'active': active, 'total': total})

# ── Admin: Wholesale buyer management ─────────────────────────────────────

@app.route('/admin/wholesale')
@login_required
def admin_wholesale():
    buyers = WholesaleBuyer.query.order_by(WholesaleBuyer.created.desc()).all()
    total_active = sum(1 for b in buyers if b.is_active)
    return render_template('wholesale/admin_buyers.html', buyers=buyers, total_active=total_active)

@app.route('/admin/wholesale/add', methods=['POST'])
@login_required
def admin_wholesale_add():
    phone    = request.form.get('phone', '').strip()
    password = request.form.get('password', '').strip()
    name     = request.form.get('name', '').strip()
    company  = request.form.get('company', '').strip()
    city     = request.form.get('city', '').strip()
    notes    = request.form.get('notes', '').strip()
    if not phone or not password:
        flash('Phone and password are required.', 'error')
        return redirect(url_for('admin_wholesale'))
    if WholesaleBuyer.query.filter_by(phone=phone).first():
        flash('A buyer with that phone number already exists.', 'error')
        return redirect(url_for('admin_wholesale'))
    buyer = WholesaleBuyer(phone=phone, name=name or None, company=company or None,
                           city=city or None, notes=notes or None)
    buyer.set_password(password)
    db.session.add(buyer)
    db.session.commit()
    flash(f'Wholesale buyer {phone} created successfully.', 'success')
    return redirect(url_for('admin_wholesale'))

@app.route('/admin/wholesale/<int:buyer_id>/toggle', methods=['POST'])
@login_required
def admin_wholesale_toggle(buyer_id):
    buyer = WholesaleBuyer.query.get_or_404(buyer_id)
    buyer.is_active = not buyer.is_active
    db.session.commit()
    status = 'enabled' if buyer.is_active else 'disabled'
    flash(f'Buyer {buyer.phone} {status}.', 'success')
    return redirect(url_for('admin_wholesale'))

@app.route('/admin/wholesale/<int:buyer_id>/delete', methods=['POST'])
@login_required
def admin_wholesale_delete(buyer_id):
    buyer = WholesaleBuyer.query.get_or_404(buyer_id)
    phone = buyer.phone
    db.session.delete(buyer)
    db.session.commit()
    flash(f'Buyer {phone} removed.', 'success')
    return redirect(url_for('admin_wholesale'))

@app.route('/admin/wholesale/<int:buyer_id>/reset-password', methods=['POST'])
@login_required
def admin_wholesale_reset_pw(buyer_id):
    buyer    = WholesaleBuyer.query.get_or_404(buyer_id)
    password = request.form.get('password', '').strip()
    if not password:
        flash('New password cannot be empty.', 'error')
        return redirect(url_for('admin_wholesale'))
    buyer.set_password(password)
    db.session.commit()
    flash(f'Password updated for {buyer.phone}.', 'success')
    return redirect(url_for('admin_wholesale'))

# ---------- DB Init ----------
with app.app_context():
    retries = 10
    while retries > 0:
        try:
            db.engine.connect()
            break
        except Exception as e:
            retries -= 1
            if retries == 0:
                raise e
            time.sleep(3)
    db.create_all()
    # Migration: add columns that may be missing from older DB schemas
    with db.engine.connect() as conn:
        try:
            conn.execute(db.text("ALTER TABLE category ADD COLUMN IF NOT EXISTS image VARCHAR(200)"))
            conn.commit()
        except Exception:
            conn.rollback()
    with db.engine.connect() as conn:
        try:
            conn.execute(db.text("ALTER TABLE category ADD COLUMN IF NOT EXISTS parent_id INTEGER REFERENCES category(id)"))
            conn.commit()
        except Exception:
            conn.rollback()
    with db.engine.connect() as conn:
        try:
            conn.execute(db.text("ALTER TABLE product ADD COLUMN IF NOT EXISTS color VARCHAR(200)"))
            conn.execute(db.text("ALTER TABLE product ADD COLUMN IF NOT EXISTS buying_for VARCHAR(200)"))
            conn.execute(db.text("ALTER TABLE product ADD COLUMN IF NOT EXISTS is_in_stock BOOLEAN NOT NULL DEFAULT TRUE"))
            conn.execute(db.text("ALTER TABLE product ADD COLUMN IF NOT EXISTS sort_order INTEGER NOT NULL DEFAULT 0"))
            conn.commit()
        except Exception:
            conn.rollback()
    with db.engine.connect() as conn:
        try:
            conn.execute(db.text("ALTER TABLE category ADD COLUMN IF NOT EXISTS description TEXT"))
            conn.commit()
        except Exception:
            conn.rollback()
    with db.engine.connect() as conn:
        try:
            conn.execute(db.text("ALTER TABLE product ADD COLUMN IF NOT EXISTS rating_value NUMERIC(3,2) DEFAULT NULL"))
            conn.execute(db.text("ALTER TABLE product ADD COLUMN IF NOT EXISTS rating_count INTEGER DEFAULT 0"))
            conn.commit()
        except Exception:
            conn.rollback()
    with db.engine.connect() as conn:
        try:
            conn.execute(db.text("ALTER TABLE product ADD COLUMN IF NOT EXISTS silver_pricing_enabled BOOLEAN NOT NULL DEFAULT FALSE"))
            conn.execute(db.text("ALTER TABLE product ADD COLUMN IF NOT EXISTS silver_weight_grams NUMERIC(8,3)"))
            conn.execute(db.text("ALTER TABLE product ADD COLUMN IF NOT EXISTS silver_multiplier NUMERIC(6,4)"))
            conn.execute(db.text("ALTER TABLE product ADD COLUMN IF NOT EXISTS silver_fixed_addition NUMERIC(10,2)"))
            conn.commit()
        except Exception:
            conn.rollback()
    Category.seed_defaults()
    Category.seed_gifts()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=os.environ.get('FLASK_ENV') != 'production')

# ═══════════════════════════════════════════════════════════════════════
#  CATALOG POPULATION API
#  All endpoints require header:  X-API-Key: <CATALOG_API_KEY env var>
#  Set CATALOG_API_KEY in your .env / Railway environment variables.
# ═══════════════════════════════════════════════════════════════════════

CATALOG_API_KEY = os.environ.get('CATALOG_API_KEY', '')

def api_key_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not CATALOG_API_KEY:
            return jsonify({'error': 'API not enabled. Set CATALOG_API_KEY environment variable.'}), 503
        key = request.headers.get('X-API-Key', '')
        if key != CATALOG_API_KEY:
            return jsonify({'error': 'Unauthorized. Invalid or missing X-API-Key header.'}), 401
        return f(*args, **kwargs)
    return decorated


# ── GET /api/v1/products ─────────────────────────────────────────────
@app.route('/api/v1/products', methods=['GET'])
@api_key_required
def apiv1_products_list():
    """List all products. Optional ?category=rings&in_stock=1"""
    q            = request.args.get('q', '').strip()
    category     = request.args.get('category', '').strip()
    in_stock     = request.args.get('in_stock') == '1'
    page         = max(1, request.args.get('page', 1, type=int))
    per_page     = min(100, request.args.get('per_page', 50, type=int))

    query = Product.query.join(Category, isouter=True)
    if q:
        query = query.filter(or_(Product.name.ilike(f'%{q}%'), Product.tags.ilike(f'%{q}%')))
    if category:
        cat = Category.query.filter(func.lower(Category.name) == category.lower()).first()
        if cat:
            sub_ids = [s.id for s in cat.subcategories.all()]
            query = query.filter(Product.category_id.in_([cat.id] + sub_ids))
    if in_stock:
        query = query.filter(Product.is_in_stock == True)

    total   = query.count()
    products = query.offset((page - 1) * per_page).limit(per_page).all()
    return jsonify({
        'total': total, 'page': page, 'per_page': per_page,
        'pages': max(1, (total + per_page - 1) // per_page),
        'products': [p.to_dict() for p in products]
    })


# ── GET /api/v1/products/<id> ────────────────────────────────────────
@app.route('/api/v1/products/<int:product_id>', methods=['GET'])
@api_key_required
def apiv1_product_get(product_id):
    p = Product.query.get_or_404(product_id)
    return jsonify(p.to_dict())


# ── POST /api/v1/products ────────────────────────────────────────────
@app.route('/api/v1/products', methods=['POST'])
@api_key_required
def apiv1_product_create():
    """
    Create a new product.
    Body (JSON):
    {
        "name": "Sterling Silver Ring",           # required
        "price": 1499.00,                         # required
        "category": "Rings",                      # name or id — optional
        "description": "...",
        "specs": {"Material": "925 Silver", "Weight": "4.5g", "Size": "7"},
        "tags": "ring,silver,solitaire",
        "synonyms": "band,ring",
        "meta_title": "...",
        "meta_description": "...",
        "meta_keywords": "...",
        "color": "Silver",
        "buying_for": "Womens",
        "is_in_stock": true,
        "sort_order": 0,
        "silver_pricing_enabled": false,
        "silver_weight_grams": 4.5,
        "silver_multiplier": 1.15,
        "silver_fixed_addition": 200
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'JSON body required'}), 400
    if not data.get('name') or data.get('price') is None:
        return jsonify({'error': '"name" and "price" are required fields'}), 400

    name = data['name'].strip()
    slug = slugify(name)
    # ensure unique slug
    existing = Product.query.filter_by(slug=slug).first()
    if existing:
        slug = f"{slug}-{secrets.token_hex(3)}"

    # resolve category
    category_id = None
    cat_val = data.get('category')
    if cat_val:
        if isinstance(cat_val, int):
            category_id = cat_val
        else:
            cat = Category.query.filter(func.lower(Category.name) == str(cat_val).lower()).first()
            if cat:
                category_id = cat.id

    p = Product(
        name                  = name,
        slug                  = slug,
        price                 = float(data['price']),
        description           = data.get('description', ''),
        category_id           = category_id,
        specs                 = data.get('specs', {}),
        tags                  = (data.get('tags', '') or '')[:500],
        synonyms              = (data.get('synonyms', '') or '')[:500],
        meta_title            = (data.get('meta_title', '') or '')[:60],
        meta_description      = (data.get('meta_description', '') or '')[:160],
        meta_keywords         = (data.get('meta_keywords', '') or '')[:200],
        color                 = data.get('color'),
        buying_for            = data.get('buying_for'),
        is_in_stock           = bool(data.get('is_in_stock', True)),
        sort_order            = int(data.get('sort_order', 0) or 0),
        rating_value          = data.get('rating_value'),
        rating_count          = int(data.get('rating_count', 0) or 0),
        silver_pricing_enabled= bool(data.get('silver_pricing_enabled', False)),
        silver_weight_grams   = data.get('silver_weight_grams'),
        silver_multiplier     = data.get('silver_multiplier'),
        silver_fixed_addition = data.get('silver_fixed_addition'),
        images                = data.get('images', []),
    )

    # auto-compute price from silver rate if enabled
    if p.silver_pricing_enabled and p.silver_weight_grams:
        live_rate = float(Settings.get('silver_live_rate_per_kg', 0) or 0)
        premium   = float(Settings.get('silver_premium_per_kg', 0) or 0)
        computed  = compute_silver_price(p, live_rate, premium)
        if computed:
            p.price = computed

    db.session.add(p)
    db.session.commit()
    return jsonify({'created': True, 'product': p.to_dict()}), 201


# ── PUT /api/v1/products/<id> ────────────────────────────────────────
@app.route('/api/v1/products/<int:product_id>', methods=['PUT', 'PATCH'])
@api_key_required
def apiv1_product_update(product_id):
    """
    Update an existing product. Send only the fields you want to change.
    Same field names as POST. PATCH and PUT both do partial update.
    """
    p = Product.query.get_or_404(product_id)
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    if 'name' in data:
        p.name = data['name'].strip()
        p.slug = slugify(p.name)
        existing = Product.query.filter(Product.slug == p.slug, Product.id != p.id).first()
        if existing:
            p.slug = f"{p.slug}-{secrets.token_hex(3)}"

    if 'price' in data:          p.price = float(data['price'])
    if 'description' in data:    p.description = data['description']
    if 'specs' in data:          p.specs = data['specs']
    if 'tags' in data:           p.tags = (data['tags'] or '')[:500]
    if 'synonyms' in data:       p.synonyms = (data['synonyms'] or '')[:500]
    if 'meta_title' in data:     p.meta_title = (data['meta_title'] or '')[:60]
    if 'meta_description' in data: p.meta_description = (data['meta_description'] or '')[:160]
    if 'meta_keywords' in data:  p.meta_keywords = (data['meta_keywords'] or '')[:200]
    if 'color' in data:          p.color = data['color']
    if 'buying_for' in data:     p.buying_for = data['buying_for']
    if 'is_in_stock' in data:    p.is_in_stock = bool(data['is_in_stock'])
    if 'sort_order' in data:     p.sort_order = int(data['sort_order'] or 0)
    if 'rating_value' in data:   p.rating_value = data['rating_value']
    if 'rating_count' in data:   p.rating_count = int(data['rating_count'] or 0)
    if 'images' in data:         p.images = data['images']

    if 'silver_pricing_enabled' in data: p.silver_pricing_enabled = bool(data['silver_pricing_enabled'])
    if 'silver_weight_grams' in data:    p.silver_weight_grams = data['silver_weight_grams']
    if 'silver_multiplier' in data:      p.silver_multiplier = data['silver_multiplier']
    if 'silver_fixed_addition' in data:  p.silver_fixed_addition = data['silver_fixed_addition']

    if 'category' in data:
        cat_val = data['category']
        if cat_val is None:
            p.category_id = None
        elif isinstance(cat_val, int):
            p.category_id = cat_val
        else:
            cat = Category.query.filter(func.lower(Category.name) == str(cat_val).lower()).first()
            p.category_id = cat.id if cat else p.category_id

    p.updated = datetime.utcnow()
    db.session.commit()
    return jsonify({'updated': True, 'product': p.to_dict()})


# ── DELETE /api/v1/products/<id> ─────────────────────────────────────
@app.route('/api/v1/products/<int:product_id>', methods=['DELETE'])
@api_key_required
def apiv1_product_delete(product_id):
    p = Product.query.get_or_404(product_id)
    name = p.name
    db.session.delete(p)
    db.session.commit()
    return jsonify({'deleted': True, 'name': name})


# ── POST /api/v1/products/bulk ───────────────────────────────────────
@app.route('/api/v1/products/bulk', methods=['POST'])
@api_key_required
def apiv1_products_bulk():
    """
    Create or update multiple products in one call.
    Body: { "products": [ {...}, {...} ] }
    Each product: same fields as POST /api/v1/products.
    If a product with the same name already exists it will be UPDATED,
    otherwise CREATED. Returns counts of each.
    """
    data = request.get_json(silent=True)
    if not data or 'products' not in data:
        return jsonify({'error': 'Body must be {"products": [...]}'}), 400

    created_count = 0
    updated_count = 0
    errors = []

    for i, item in enumerate(data['products']):
        try:
            if not item.get('name') or item.get('price') is None:
                errors.append({'index': i, 'error': 'name and price required'})
                continue

            name = item['name'].strip()
            existing = Product.query.filter(
                func.lower(Product.name) == name.lower()
            ).first()

            # resolve category
            category_id = None
            cat_val = item.get('category')
            if cat_val:
                if isinstance(cat_val, int):
                    category_id = cat_val
                else:
                    cat = Category.query.filter(
                        func.lower(Category.name) == str(cat_val).lower()
                    ).first()
                    if cat:
                        category_id = cat.id

            if existing:
                # update
                p = existing
                p.price       = float(item['price'])
                p.description = item.get('description', p.description)
                p.category_id = category_id if cat_val is not None else p.category_id
                p.specs       = item.get('specs', p.specs)
                p.tags        = (item.get('tags', p.tags) or '')[:500]
                p.synonyms    = (item.get('synonyms', p.synonyms) or '')[:500]
                p.is_in_stock = bool(item.get('is_in_stock', p.is_in_stock))
                p.color       = item.get('color', p.color)
                p.buying_for  = item.get('buying_for', p.buying_for)
                p.sort_order  = int(item.get('sort_order', p.sort_order) or 0)
                if 'silver_weight_grams' in item:
                    p.silver_weight_grams = item['silver_weight_grams']
                if 'silver_multiplier' in item:
                    p.silver_multiplier = item['silver_multiplier']
                if 'silver_pricing_enabled' in item:
                    p.silver_pricing_enabled = bool(item['silver_pricing_enabled'])
                if 'images' in item:
                    p.images = item['images']
                p.updated = datetime.utcnow()
                updated_count += 1
            else:
                # create
                slug = slugify(name)
                if Product.query.filter_by(slug=slug).first():
                    slug = f"{slug}-{secrets.token_hex(3)}"
                p = Product(
                    name                  = name,
                    slug                  = slug,
                    price                 = float(item['price']),
                    description           = item.get('description', ''),
                    category_id           = category_id,
                    specs                 = item.get('specs', {}),
                    tags                  = (item.get('tags', '') or '')[:500],
                    synonyms              = (item.get('synonyms', '') or '')[:500],
                    meta_title            = (item.get('meta_title', '') or '')[:60],
                    meta_description      = (item.get('meta_description', '') or '')[:160],
                    meta_keywords         = (item.get('meta_keywords', '') or '')[:200],
                    color                 = item.get('color'),
                    buying_for            = item.get('buying_for'),
                    is_in_stock           = bool(item.get('is_in_stock', True)),
                    sort_order            = int(item.get('sort_order', 0) or 0),
                    silver_pricing_enabled= bool(item.get('silver_pricing_enabled', False)),
                    silver_weight_grams   = item.get('silver_weight_grams'),
                    silver_multiplier     = item.get('silver_multiplier'),
                    silver_fixed_addition = item.get('silver_fixed_addition'),
                    images                = item.get('images', []),
                )
                db.session.add(p)
                created_count += 1
        except Exception as e:
            errors.append({'index': i, 'name': item.get('name'), 'error': str(e)})

    db.session.commit()
    return jsonify({
        'created': created_count,
        'updated': updated_count,
        'errors':  errors,
        'total_processed': created_count + updated_count + len(errors)
    }), 200 if not errors else 207


# ── GET /api/v1/categories ───────────────────────────────────────────
@app.route('/api/v1/categories', methods=['GET'])
@api_key_required
def apiv1_categories_list():
    cats = Category.query.order_by(Category.name).all()
    return jsonify([{
        'id': c.id, 'name': c.name, 'slug': c.slug,
        'parent_id': c.parent_id,
        'product_count': len(c.products)
    } for c in cats])


# ── POST /api/v1/categories ──────────────────────────────────────────
@app.route('/api/v1/categories', methods=['POST'])
@api_key_required
def apiv1_category_create():
    """Body: {"name": "Rings", "parent": "Gifts"}  (parent is optional)"""
    data = request.get_json(silent=True)
    if not data or not data.get('name'):
        return jsonify({'error': '"name" is required'}), 400
    name = data['name'].strip()
    slug = slugify(name)
    if Category.query.filter_by(slug=slug).first():
        return jsonify({'error': f'Category "{name}" already exists'}), 409
    parent_id = None
    if data.get('parent'):
        parent = Category.query.filter(
            func.lower(Category.name) == str(data['parent']).lower()
        ).first()
        if parent:
            parent_id = parent.id
    cat = Category(name=name, slug=slug, spec_schema=[], parent_id=parent_id)
    db.session.add(cat)
    db.session.commit()
    return jsonify({'created': True, 'id': cat.id, 'name': cat.name, 'slug': cat.slug}), 201


# ── POST /api/v1/settings/silver-rate ───────────────────────────────
@app.route('/api/v1/settings/silver-rate', methods=['GET', 'POST'])
@api_key_required
def apiv1_silver_rate():
    """
    GET  — return current silver rate settings
    POST — set silver rate, premium, and optionally auto-update all silver-priced products

    Body (POST):
    {
        "live_rate_per_kg": 235000,   # required — live silver spot rate in ₹/kg
        "premium_per_kg":   50000,    # optional — your making/premium charge in ₹/kg
        "auto_update":      true      # optional — if true, recomputes price for all
                                      #            silver-priced products immediately
    }
    """
    if request.method == 'GET':
        return jsonify({
            'live_rate_per_kg': Settings.get('silver_live_rate_per_kg', 0),
            'premium_per_kg':   Settings.get('silver_premium_per_kg',   0),
            'auto_update':      Settings.get('silver_auto_update',       False),
            'updated_at':       Settings.get('silver_rate_updated_at',   None),
        })

    data = request.get_json(silent=True)
    if not data or data.get('live_rate_per_kg') is None:
        return jsonify({'error': '"live_rate_per_kg" is required'}), 400

    live_rate   = float(data['live_rate_per_kg'])
    premium     = float(data.get('premium_per_kg', 0) or 0)
    auto_update = bool(data.get('auto_update', False))

    Settings.set('silver_live_rate_per_kg', live_rate)
    Settings.set('silver_premium_per_kg',   premium)
    Settings.set('silver_auto_update',       auto_update)
    Settings.set('silver_rate_updated_at',   datetime.utcnow().isoformat())

    updated_products = 0
    if auto_update:
        products = Product.query.filter_by(silver_pricing_enabled=True).all()
        for p in products:
            new_price = compute_silver_price(p, live_rate, premium)
            if new_price is not None:
                p.price = new_price
                updated_products += 1
        db.session.commit()

    return jsonify({
        'ok': True,
        'live_rate_per_kg':   live_rate,
        'premium_per_kg':     premium,
        'auto_update':        auto_update,
        'updated_products':   updated_products,
        'updated_at':         Settings.get('silver_rate_updated_at'),
    })