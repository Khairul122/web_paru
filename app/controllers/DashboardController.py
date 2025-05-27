from flask import Blueprint, render_template, session, redirect, url_for, jsonify
from datetime import datetime, timedelta
from app.models.AuthModel import User
from app.extension import db

try:
    from app.models.DatasetModel import DatasetGambar
except:
    DatasetGambar = None

try:
    from app.models.KategoriModel import KategoriPenyakit
except:
    KategoriPenyakit = None

try:
    from app.models.DataCitraModel import DataCitra
except:
    DataCitra = None

try:
    from app.models.SplitDataModel import SplitData
except:
    SplitData = None

try:
    from app.models.LDAModel import HasilKlasifikasi
except:
    HasilKlasifikasi = None

dashboard_bp = Blueprint('dashboard', __name__)

def get_stats():
    stats = {
        'total_users': 0,
        'total_categories': 0,
        'total_datasets': 0,
        'total_features': 0,
        'train_data': 0,
        'test_data': 0,
        'total_classifications': 0,
        'category_distribution': [],
        'classification_distribution': [],
        'recent_classifications': []
    }
    
    try:
        stats['total_users'] = User.query.count()
        
        if KategoriPenyakit:
            stats['total_categories'] = KategoriPenyakit.query.count()
        
        if DatasetGambar:
            stats['total_datasets'] = DatasetGambar.query.count()
        
        if DataCitra:
            stats['total_features'] = DataCitra.query.count()
        
        if SplitData:
            stats['train_data'] = SplitData.query.filter_by(jenis_split='train').count()
            stats['test_data'] = SplitData.query.filter_by(jenis_split='test').count()
        
        if HasilKlasifikasi:
            stats['total_classifications'] = HasilKlasifikasi.query.count()
        
        if KategoriPenyakit and DatasetGambar:
            try:
                from sqlalchemy import func
                category_dist = db.session.query(
                    KategoriPenyakit.nama_kategori,
                    func.count(DatasetGambar.id_gambar).label('count')
                ).outerjoin(
                    DatasetGambar, KategoriPenyakit.id_kategori == DatasetGambar.id_kategori
                ).group_by(KategoriPenyakit.nama_kategori).all()
                stats['category_distribution'] = category_dist
            except:
                pass
        
        if KategoriPenyakit and HasilKlasifikasi:
            try:
                from sqlalchemy import func
                classification_dist = db.session.query(
                    KategoriPenyakit.nama_kategori,
                    func.count(HasilKlasifikasi.id_hasil).label('count')
                ).join(
                    HasilKlasifikasi, KategoriPenyakit.id_kategori == HasilKlasifikasi.id_kategori
                ).group_by(KategoriPenyakit.nama_kategori).all()
                stats['classification_distribution'] = classification_dist
            except:
                pass
        
        if HasilKlasifikasi and DataCitra and DatasetGambar and KategoriPenyakit:
            try:
                from sqlalchemy import desc
                recent = db.session.query(
                    HasilKlasifikasi, DatasetGambar, KategoriPenyakit
                ).join(
                    DataCitra, HasilKlasifikasi.id_data_citra == DataCitra.id_data_citra
                ).join(
                    DatasetGambar, DataCitra.id_gambar == DatasetGambar.id_gambar
                ).join(
                    KategoriPenyakit, HasilKlasifikasi.id_kategori == KategoriPenyakit.id_kategori
                ).order_by(desc(HasilKlasifikasi.tanggal_klasifikasi)).limit(5).all()
                stats['recent_classifications'] = recent
            except:
                pass
    except:
        pass
    
    return stats

@dashboard_bp.route('/')
@dashboard_bp.route('/dashboard')
def index():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))

    user = User.query.get(session['user_id'])
    if not user:
        return redirect(url_for('auth.login'))

    now = datetime.now() + timedelta(hours=7)
    current_hour = now.hour

    if 5 <= current_hour < 12:
        greeting = "Selamat Pagi"
        greeting_icon = "fas fa-sun"
    elif 12 <= current_hour < 15:
        greeting = "Selamat Siang"
        greeting_icon = "fas fa-sun"
    elif 15 <= current_hour < 18:
        greeting = "Selamat Sore"
        greeting_icon = "fas fa-cloud-sun"
    else:
        greeting = "Selamat Malam"
        greeting_icon = "fas fa-moon"

    stats = get_stats()

    return render_template(
        'views/dashboard/index.html',
        user=user,
        greeting=greeting,
        greeting_icon=greeting_icon,
        user_name=user.nama_lengkap,
        current_time=now,
        stats=stats
    )

@dashboard_bp.route('/api/stats')
def api_stats():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        stats = get_stats()
        
        return jsonify({
            'success': True,
            'stats': {
                'total_users': stats['total_users'],
                'total_categories': stats['total_categories'],
                'total_datasets': stats['total_datasets'],
                'total_features': stats['total_features'],
                'train_data': stats['train_data'],
                'test_data': stats['test_data'],
                'total_classifications': stats['total_classifications']
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@dashboard_bp.route('/test')
def test():
    try:
        stats = get_stats()
        return jsonify({
            'status': 'success',
            'message': 'Dashboard working',
            'stats': stats
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500