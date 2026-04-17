import os, re, json, secrets, time
from datetime import datetime
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import or_, and_, func
from sqlalchemy.orm import joinedload
from werkzeug.utils import secure_filename
import google.generativeai as genai

# ---------- App & Config ----------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-change-me')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@db:5432/jewellery')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}

db = SQLAlchemy(app)
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-2.5-pro')  # Auto-selects latest stable version

# ---------- Context Processor ----------
@app.context_processor
def inject_now():
    return {'now': datetime.utcnow()}

# ---------- Helpers ----------
def slugify(text):
    return re.sub(r'[-\s]+', '-', re.sub(r'[^\w\s-]', '', text.strip().lower()))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def compress_image(filepath, max_size=(1200, 1200), quality=85):
    with Image.open(filepath) as img:
        img.thumbnail(max_size)
        img.save(filepath, optimize=True, quality=quality)

def save_images(files):
    filenames = []
    for f in files:
        if f and allowed_file(f.filename):
            ext = f.filename.rsplit('.', 1)[1].lower()
            filename = f"{secrets.token_hex(8)}.{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(filepath)
            compress_image(filepath)
            filenames.append(filename)
    return filenames

# ---------- Models ----------
class Category(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    slug = db.Column(db.String(120), unique=True, nullable=False)
    spec_schema = db.Column(db.JSON, default=list)
    image = db.Column(db.String(200), nullable=True)

    @staticmethod
    def seed_defaults():
        defaults = [
            ('Rings', ['Material', 'Weight', 'Size', 'Gemstone']),
            ('Necklaces', ['Material', 'Length', 'Clasp Type', 'Pendant']),
            ('Earrings', ['Material', 'Type', 'Length', 'Closure']),
            ('Bracelets', ['Material', 'Length', 'Width', 'Clasp']),
            ('Bangles', ['Material', 'Diameter', 'Width', 'Design']),
            ('Chains', ['Material', 'Length', 'Thickness', 'Style'])
        ]
        for name, schema in defaults:
            if not Category.query.filter_by(name=name).first():
                db.session.add(Category(name=name, slug=slugify(name), spec_schema=schema))
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
    created = db.Column(db.DateTime, default=datetime.utcnow)
    updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id, 'name': self.name, 'slug': self.slug,
            'price': float(self.price), 'description': self.description,
            'category': self.category.name if self.category else None,
            'specs': self.specs, 'images': self.images,
            'meta_title': self.meta_title, 'meta_description': self.meta_description,
            'meta_keywords': self.meta_keywords, 'tags': self.tags, 'synonyms': self.synonyms
        }

# ---------- Seed Sample Products ----------
def seed_sample_products():
    if Product.query.count() > 0:
        return
    rings = Category.query.filter_by(name='Rings').first()
    necklaces = Category.query.filter_by(name='Necklaces').first()
    earrings = Category.query.filter_by(name='Earrings').first()

    samples = [
        Product(
            name='Solitaire Diamond Ring',
            price=125000.00,
            description='Exquisite solitaire diamond set in 18K white gold.',
            category=rings,
            specs={'Material': '18K White Gold', 'Weight': '4.5g', 'Size': '7', 'Gemstone': 'Diamond (0.5ct)'},
            meta_title='Solitaire Diamond Ring | 18K White Gold',
            meta_description='Buy solitaire diamond ring in 18K white gold. 0.5ct diamond, elegant design.',
            meta_keywords='diamond ring, solitaire, white gold, engagement ring',
            tags='diamond,solitaire,engagement',
            synonyms='band,ring'
        ),
        Product(
            name='Pearl Pendant Necklace',
            price=45000.00,
            description='Cultured freshwater pearl with sterling silver chain.',
            category=necklaces,
            specs={'Material': 'Sterling Silver', 'Length': '18 inches', 'Clasp Type': 'Lobster Claw', 'Pendant': 'Freshwater Pearl'},
            meta_title='Pearl Pendant Necklace | Sterling Silver',
            meta_description='Elegant pearl pendant necklace on sterling silver chain.',
            meta_keywords='pearl necklace, pendant, sterling silver',
            tags='pearl,necklace,pendant',
            synonyms='chain,neckwear'
        ),
        Product(
            name='Gold Hoop Earrings',
            price=22000.00,
            description='Classic 22K gold hoop earrings, lightweight and timeless.',
            category=earrings,
            specs={'Material': '22K Gold', 'Type': 'Hoop', 'Length': '2.5cm', 'Closure': 'Lever Back'},
            meta_title='Gold Hoop Earrings | 22K Yellow Gold',
            meta_description='Shop classic 22K gold hoop earrings. Lightweight and perfect for daily wear.',
            meta_keywords='gold earrings, hoop earrings, 22K gold',
            tags='gold,hoops,earrings',
            synonyms='hoop earrings,gold hoops'
        )
    ]
    for p in samples:
        p.slug = slugify(p.name)
        db.session.add(p)
    db.session.commit()

# ---------- Routes ----------
@app.route('/')
def index():
    categories = Category.query.all()
    featured = Product.query.order_by(Product.created.desc()).limit(8).all()
    return render_template('index.html', categories=categories, products=featured)

@app.route('/product/<slug>')
def product_detail(slug):
    product = Product.query.filter_by(slug=slug).first_or_404()
    return render_template('product.html', product=product)

@app.route('/search')
def search():
    q = request.args.get('q', '').strip()
    products = []
    if q:
        # Word-boundary aware: match query as a whole word to avoid
        # "rings" matching "earrings". We fetch candidates via ILIKE then
        # filter in Python using a word-boundary regex.
        import re as _re
        pattern = _re.compile(r'(?<![a-zA-Z])' + _re.escape(q) + r'(?![a-zA-Z])', _re.IGNORECASE)

        candidates = Product.query.join(Category, isouter=True).filter(
            or_(
                Product.name.ilike(f'%{q}%'),
                Product.description.ilike(f'%{q}%'),
                Product.tags.ilike(f'%{q}%'),
                Product.synonyms.ilike(f'%{q}%'),
                Product.meta_keywords.ilike(f'%{q}%'),
                Category.name.ilike(f'%{q}%'),
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
        ]
    return render_template('search.html', query=q, products=products)

# ---------- Admin ----------
@app.route('/admin')
def admin_dashboard():
    categories = Category.query.options(joinedload(Category.products)).all()
    return render_template('admin/dashboard.html', categories=categories)

@app.route('/admin/product/new', methods=['GET', 'POST'])
@app.route('/admin/product/<int:id>/edit', methods=['GET', 'POST'])
def admin_product_form(id=None):
    product = Product.query.get(id) if id else None
    categories = Category.query.all()
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

        if 'images' in request.files:
            new_images = save_images(request.files.getlist('images'))
            kept_images = request.form.getlist('images_keep')
            # kept_images preserves the drag-sorted order from the form;
            # new images are appended after (their order matches the file input order)
            product.images = kept_images + new_images if (kept_images or new_images) else []
        db.session.commit()
        flash('Product saved.', 'success')
        return redirect(url_for('admin_dashboard'))
    return render_template('admin/product_form.html', product=product, categories=categories)

@app.route('/admin/product/<int:id>/delete', methods=['POST'])
def admin_product_delete(id):
    product = Product.query.get_or_404(id)
    db.session.delete(product)
    db.session.commit()
    flash('Product deleted.', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/categories', methods=['GET', 'POST'])
def admin_categories():
    if request.method == 'POST':
        name = request.form['name']
        slug = slugify(name)
        if not Category.query.filter_by(slug=slug).first():
            db.session.add(Category(name=name, slug=slug, spec_schema=[]))
            db.session.commit()
            flash('Category added.', 'success')
        else:
            flash('Category already exists.', 'warning')
        return redirect(url_for('admin_categories'))
    categories = Category.query.all()
    return render_template('admin/categories.html', categories=categories)

@app.route('/admin/category/<int:id>/delete', methods=['POST'])
def admin_category_delete(id):
    cat = Category.query.get_or_404(id)
    db.session.delete(cat)
    db.session.commit()
    flash('Category deleted.', 'success')
    return redirect(url_for('admin_categories'))
@app.route('/admin/category/<int:id>/edit', methods=['POST'])
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
        db.session.commit()
        flash('Category renamed.', 'success')
    return redirect(url_for('admin_categories'))

@app.route('/admin/category/<int:id>/upload-image', methods=['POST'])
def admin_category_upload_image(id):
    cat = Category.query.get_or_404(id)
    f = request.files.get('image')
    if f and allowed_file(f.filename):
        ext = f.filename.rsplit('.', 1)[1].lower()
        filename = f"cat_{secrets.token_hex(6)}.{ext}"

        # Ensure assets/categories directory exists
        cat_upload_dir = os.path.join('templates', 'assets', 'categories')
        os.makedirs(cat_upload_dir, exist_ok=True)

        filepath = os.path.join(cat_upload_dir, filename)
        f.save(filepath)
        compress_image(filepath)
        cat.image = filename
        db.session.commit()
        flash('Category image updated.', 'success')
    else:
        flash('Invalid file.', 'warning')
    return redirect(url_for('admin_categories'))

# ---------- AJAX Endpoints for Category & AI Specs ----------
@app.route('/admin/category/add-ajax', methods=['POST'])
def add_category_ajax():
    name = request.json.get('name', '').strip()
    if not name:
        return jsonify({'error': 'Name required'}), 400
    slug = slugify(name)
    if Category.query.filter_by(slug=slug).first():
        return jsonify({'error': 'Category already exists'}), 400
    cat = Category(name=name, slug=slug, spec_schema=[])
    db.session.add(cat)
    db.session.commit()
    return jsonify({'id': cat.id, 'name': cat.name})

@app.route('/admin/ai-suggest-specs', methods=['POST'])
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

# ---------- AI Endpoints (SEO & Assistant) ----------
@app.route('/admin/ai-seo', methods=['POST'])
def ai_seo():
    data = request.json
    name = data.get('name', '')
    desc = data.get('desc', '')
    prompt = f"""Generate SEO for jewellery: name="{name}", desc="{desc}". Output JSON with keys: meta_title(max60), meta_description(max160), meta_keywords(comma), tags(comma), synonyms(comma). Only JSON."""
    try:
        resp = model.generate_content(prompt, generation_config={"temperature":0.2})
        text = resp.text.strip().replace('```json','').replace('```','')
        seo = json.loads(text)
        return jsonify(seo)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/ai-assistant', methods=['POST'])
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
        resp = model.generate_content(prompt, generation_config={"temperature":0.0})
        text = resp.text.strip().replace('```json','').replace('```','')
        action = json.loads(text)
        return jsonify(action)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/apply-action', methods=['POST'])
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

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/assets/<path:filename>')
def assets_file(filename):
    return send_from_directory(os.path.join('templates', 'assets'), filename)

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
    Category.seed_defaults()
    seed_sample_products()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)