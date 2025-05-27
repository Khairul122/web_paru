from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify, make_response
from app.extension import db
from app.models.DataCitraModel import DataCitra
from app.models.DatasetModel import DatasetGambar
from app.models.KategoriModel import KategoriPenyakit
from app.models.SplitDataModel import SplitData
from app.models.LDAModel import HasilKlasifikasi
from sqlalchemy import func, desc, asc
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import csv
import io
import sys
from datetime import datetime

lda_bp = Blueprint('lda', __name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models', 'saved_models')

def simple_log(message):
    """Logging sederhana untuk debug"""
    print(f"[LDA DEBUG] {message}")
    sys.stdout.flush()

def fix_image_path(path_file):
    """Perbaiki path gambar"""
    if not path_file:
        return 'img/no-image.png'
    image_path = path_file.replace('\\', '/')
    if not image_path.startswith('static/'):
        if 'bronkitis' in image_path:
            image_path = f"static/uploads/dataset/bronkitis/{os.path.basename(image_path)}"
        elif 'tuberkolosis' in image_path:
            image_path = f"static/uploads/dataset/tuberkolosis/{os.path.basename(image_path)}"
        elif 'pneumonia' in image_path:
            image_path = f"static/uploads/dataset/pneumonia/{os.path.basename(image_path)}"
        elif 'normal' in image_path:
            image_path = f"static/uploads/dataset/normal/{os.path.basename(image_path)}"
    return image_path

def get_pagination_info(page, per_page, total):
    """Informasi pagination"""
    total_pages = (total + per_page - 1) // per_page
    has_prev = page > 1
    has_next = page < total_pages
    prev_num = page - 1 if has_prev else None
    next_num = page + 1 if has_next else None
    
    return {
        'page': page,
        'per_page': per_page,
        'total': total,
        'total_pages': total_pages,
        'has_prev': has_prev,
        'has_next': has_next,
        'prev_num': prev_num,
        'next_num': next_num
    }

def prepare_training_data():
    """Persiapan data training - DIPERBAIKI"""
    try:
        simple_log("=== MULAI PERSIAPAN DATA TRAINING ===")

        train_data = db.session.query(
            SplitData.id_data_citra,
            SplitData.id_kategori,
            DataCitra.contrast,
            DataCitra.dissimilarity,
            DataCitra.homogeneity,
            DataCitra.energy,
            DataCitra.correlation,
            DataCitra.asm,
            KategoriPenyakit.nama_kategori
        ).join(
            DataCitra, SplitData.id_data_citra == DataCitra.id_data_citra
        ).join(
            KategoriPenyakit, SplitData.id_kategori == KategoriPenyakit.id_kategori
        ).filter(
            SplitData.jenis_split == 'train'
        ).all()
        
        simple_log(f"Query hasil: {len(train_data)} records")
        
        if not train_data:
            simple_log("TIDAK ADA DATA TRAINING!")
            return None, None, None

        features = []
        labels = []
        
        for row in train_data:
            feature_values = [
                float(row.contrast or 0),
                float(row.dissimilarity or 0),
                float(row.homogeneity or 0),
                float(row.energy or 0),
                float(row.correlation or 0),
                float(row.asm or 0)
            ]
            
            features.append(feature_values)
            labels.append(row.nama_kategori)
            
            simple_log(f"Data {len(features)}: {row.nama_kategori} - Features: {feature_values}")
        
        X = np.array(features)
        y = np.array(labels)
        unique_labels = list(set(labels))
        
        simple_log(f"Training data shape: {X.shape}")
        simple_log(f"Unique labels: {unique_labels}")
        simple_log(f"Label counts: {np.unique(y, return_counts=True)}")
        
        return X, y, unique_labels
        
    except Exception as e:
        simple_log(f"ERROR dalam prepare_training_data: {str(e)}")
        return None, None, None

def prepare_test_data():
    """Persiapan data testing - DIPERBAIKI"""
    try:
        simple_log("=== MULAI PERSIAPAN DATA TESTING ===")
        
        # Query yang lebih sederhana
        test_data = db.session.query(
            SplitData.id_data_citra,
            SplitData.id_kategori,
            DataCitra.contrast,
            DataCitra.dissimilarity,
            DataCitra.homogeneity,
            DataCitra.energy,
            DataCitra.correlation,
            DataCitra.asm,
            KategoriPenyakit.nama_kategori
        ).join(
            DataCitra, SplitData.id_data_citra == DataCitra.id_data_citra
        ).join(
            KategoriPenyakit, SplitData.id_kategori == KategoriPenyakit.id_kategori
        ).filter(
            SplitData.jenis_split == 'test'
        ).all()
        
        simple_log(f"Query hasil: {len(test_data)} records")
        
        if not test_data:
            simple_log("TIDAK ADA DATA TESTING!")
            return None, None, None, None
        
        features = []
        labels = []
        data_ids = []
        
        for row in test_data:
            feature_values = [
                float(row.contrast or 0),
                float(row.dissimilarity or 0),
                float(row.homogeneity or 0),
                float(row.energy or 0),
                float(row.correlation or 0),
                float(row.asm or 0)
            ]
            
            features.append(feature_values)
            labels.append(row.nama_kategori)
            data_ids.append(row.id_data_citra)
        
        X = np.array(features)
        y = np.array(labels)
        
        simple_log(f"Test data shape: {X.shape}")
        
        return X, y, data_ids, None
        
    except Exception as e:
        simple_log(f"ERROR dalam prepare_test_data: {str(e)}")
        return None, None, None, None

def get_lda_statistics():
    """Statistik LDA"""
    try:
        total_klasifikasi = HasilKlasifikasi.query.count()
        train_count = db.session.query(SplitData).filter_by(jenis_split='train').count()
        test_count = db.session.query(SplitData).filter_by(jenis_split='test').count()
        
        kategori_stats = db.session.query(
            KategoriPenyakit.nama_kategori,
            func.count(HasilKlasifikasi.id_hasil).label('total')
        ).join(
            HasilKlasifikasi, KategoriPenyakit.id_kategori == HasilKlasifikasi.id_kategori
        ).group_by(
            KategoriPenyakit.nama_kategori
        ).all()
        
        latest_model = None
        model_path = os.path.join(MODEL_DIR, 'lda_model.pkl')
        if os.path.exists(model_path):
            latest_model = datetime.fromtimestamp(os.path.getmtime(model_path))
        
        return {
            'total_klasifikasi': total_klasifikasi,
            'train_count': train_count,
            'test_count': test_count,
            'kategori_stats': kategori_stats,
            'latest_model': latest_model
        }
    except Exception as e:
        simple_log(f"ERROR get_lda_statistics: {str(e)}")
        return {
            'total_klasifikasi': 0,
            'train_count': 0,
            'test_count': 0,
            'kategori_stats': [],
            'latest_model': None
        }

@lda_bp.route('/')
def index():
    """Halaman utama LDA"""
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    kategori_filter = request.args.get('kategori', '', type=str)
    sort_by = request.args.get('sort', 'tanggal_klasifikasi', type=str)
    sort_order = request.args.get('order', 'desc', type=str)
    
    if per_page not in [5, 10, 25, 50, 100]:
        per_page = 10
    
    query = db.session.query(
        HasilKlasifikasi, DataCitra, DatasetGambar, KategoriPenyakit
    ).join(
        DataCitra, HasilKlasifikasi.id_data_citra == DataCitra.id_data_citra
    ).join(
        DatasetGambar, DataCitra.id_gambar == DatasetGambar.id_gambar
    ).join(
        KategoriPenyakit, HasilKlasifikasi.id_kategori == KategoriPenyakit.id_kategori
    )
    
    if kategori_filter:
        query = query.filter(KategoriPenyakit.id_kategori == kategori_filter)
    
    if sort_by == 'tanggal_klasifikasi':
        if sort_order == 'desc':
            query = query.order_by(desc(HasilKlasifikasi.tanggal_klasifikasi))
        else:
            query = query.order_by(asc(HasilKlasifikasi.tanggal_klasifikasi))
    elif sort_by == 'skor_lda':
        if sort_order == 'desc':
            query = query.order_by(desc(HasilKlasifikasi.skor_lda))
        else:
            query = query.order_by(asc(HasilKlasifikasi.skor_lda))
    elif sort_by == 'kategori':
        if sort_order == 'desc':
            query = query.order_by(desc(KategoriPenyakit.nama_kategori))
        else:
            query = query.order_by(asc(KategoriPenyakit.nama_kategori))
    
    total = query.count()
    offset = (page - 1) * per_page
    hasil_list = query.offset(offset).limit(per_page).all()
    
    processed_results = []
    for hasil, data_citra, dataset, kategori in hasil_list:
        dataset.fixed_image_path = fix_image_path(dataset.path_file)
        processed_results.append((hasil, data_citra, dataset, kategori))
    
    pagination = get_pagination_info(page, per_page, total)
    stats = get_lda_statistics()
    kategori_list = KategoriPenyakit.query.all()
    
    return render_template('views/lda/index.html',
                         hasil_list=processed_results,
                         pagination=pagination,
                         stats=stats,
                         kategori_list=kategori_list,
                         current_filters={
                             'kategori': kategori_filter,
                             'sort': sort_by,
                             'order': sort_order,
                             'per_page': per_page
                         })

@lda_bp.route('/train')
def train():
    """Halaman training"""
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    stats = get_lda_statistics()
    return render_template('views/lda/train.html', stats=stats)

@lda_bp.route('/process_train', methods=['POST'])
def process_train():
    """PROSES TRAINING - DIPERBAIKI TOTAL"""
    if 'user_id' not in session:
        flash('Silakan login terlebih dahulu', 'error')
        return redirect(url_for('auth.login'))
    
    try:
        simple_log("========================================")
        simple_log("=== MULAI PROSES TRAINING LDA ===")
        simple_log("========================================")

        simple_log("STEP 1: Mengecek data training...")
        X_train, y_train, label_names = prepare_training_data()
        
        if X_train is None or len(X_train) == 0:
            simple_log("ERROR: Tidak ada data training!")
            flash('Tidak ada data training yang tersedia. Silakan lakukan split data terlebih dahulu.', 'error')
            return redirect(url_for('lda.train'))
        
        simple_log(f"BERHASIL: {len(X_train)} data training siap")
        simple_log(f"Features shape: {X_train.shape}")
        simple_log(f"Labels: {label_names}")
        
        simple_log("STEP 2: Validasi data...")
        if len(set(y_train)) < 2:
            simple_log("ERROR: Minimal 2 kategori diperlukan untuk LDA")
            flash('Minimal 2 kategori diperlukan untuk training LDA', 'error')
            return redirect(url_for('lda.train'))

        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            simple_log("WARNING: Data mengandung NaN atau Inf, akan dibersihkan")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=0.0)
        
        simple_log("Data valid untuk training")
        
        simple_log("STEP 3: Normalisasi data...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        simple_log(f"Data berhasil dinormalisasi: {X_train_scaled.shape}")
        
        simple_log("STEP 4: Encoding labels...")
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        simple_log(f"Labels encoded: {label_encoder.classes_}")

        simple_log("STEP 5: Training model LDA...")
        lda_model = LinearDiscriminantAnalysis()
        lda_model.fit(X_train_scaled, y_train_encoded)
        simple_log("Model LDA berhasil dilatih!")

        simple_log("STEP 6: Evaluasi model...")
        y_train_pred = lda_model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train_encoded, y_train_pred)
        simple_log(f"Training accuracy: {train_accuracy:.4f}")
        
        simple_log("STEP 7: Menyimpan model...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        model_data = {
            'lda_model': lda_model,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'feature_names': ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'asm'],
            'class_names': label_names,
            'training_accuracy': train_accuracy,
            'trained_at': datetime.now()
        }
        
        model_path = os.path.join(MODEL_DIR, 'lda_model.pkl')
        joblib.dump(model_data, model_path)
        simple_log(f"Model disimpan di: {model_path}")
        
        simple_log("========================================")
        simple_log("=== TRAINING BERHASIL SELESAI! ===")
        simple_log("========================================")
        
        flash(f'Training berhasil! Model LDA telah dilatih dengan akurasi: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)', 'success')
        return redirect(url_for('lda.train'))
        
    except Exception as e:
        simple_log(f"========================================")
        simple_log(f"=== ERROR DALAM TRAINING: {str(e)} ===")
        simple_log("========================================")
        flash(f'Error dalam training: {str(e)}', 'error')
        return redirect(url_for('lda.train'))

@lda_bp.route('/predict', methods=['POST'])
def predict():
    """Prediksi menggunakan model"""
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    try:
        simple_log("=== MULAI PREDIKSI ===")
        
        model_path = os.path.join(MODEL_DIR, 'lda_model.pkl')
        
        if not os.path.exists(model_path):
            simple_log("ERROR: Model tidak ditemukan!")
            flash('Model belum dilatih. Silakan lakukan training terlebih dahulu.', 'error')
            return redirect(url_for('lda.train'))
        
        simple_log("Loading model...")
        model_data = joblib.load(model_path)
        lda_model = model_data['lda_model']
        scaler = model_data['scaler']
        label_encoder = model_data['label_encoder']
        
        X_test, y_test, data_ids, _ = prepare_test_data()
        
        if X_test is None:
            simple_log("ERROR: Tidak ada data test!")
            flash('Tidak ada data test. Silakan lakukan split data terlebih dahulu.', 'error')
            return redirect(url_for('lda.index'))
        
        simple_log("Melakukan prediksi...")
        X_test_scaled = scaler.transform(X_test)
        y_pred_encoded = lda_model.predict(X_test_scaled)
        y_pred_proba = lda_model.predict_proba(X_test_scaled)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        
        simple_log("Menyimpan hasil prediksi...")
        HasilKlasifikasi.query.delete()
        
        success_count = 0
        for i, data_id in enumerate(data_ids):
            kategori_pred = y_pred[i]
            skor_max = np.max(y_pred_proba[i])
            
            kategori_obj = KategoriPenyakit.query.filter_by(nama_kategori=kategori_pred).first()
            if kategori_obj:
                hasil = HasilKlasifikasi(
                    id_data_citra=int(data_id),
                    id_kategori=int(kategori_obj.id_kategori),
                    skor_lda=float(skor_max)
                )
                db.session.add(hasil)
                success_count += 1
        
        db.session.commit()
        
        y_test_encoded = label_encoder.transform(y_test)
        test_accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
        
        simple_log(f"=== PREDIKSI SELESAI ===")
        simple_log(f"Berhasil: {success_count}, Akurasi: {test_accuracy:.4f}")
        
        flash(f'Prediksi berhasil! {success_count} hasil disimpan dengan akurasi: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)', 'success')
        return redirect(url_for('lda.index'))
        
    except Exception as e:
        simple_log(f"ERROR dalam prediksi: {str(e)}")
        flash(f'Error dalam prediksi: {str(e)}', 'error')
        return redirect(url_for('lda.index'))

@lda_bp.route('/evaluate')
def evaluate():
    """Evaluasi model"""
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    try:
        model_path = os.path.join(MODEL_DIR, 'lda_model.pkl')
        
        if not os.path.exists(model_path):
            flash('Model belum dilatih', 'error')
            return redirect(url_for('lda.train'))
        
        model_data = joblib.load(model_path)
        lda_model = model_data['lda_model']
        scaler = model_data['scaler']
        label_encoder = model_data['label_encoder']
        
        X_test, y_test, data_ids, _ = prepare_test_data()
        
        if X_test is None:
            flash('Tidak ada data test', 'error')
            return redirect(url_for('lda.index'))
        
        X_test_scaled = scaler.transform(X_test)
        y_pred_encoded = lda_model.predict(X_test_scaled)
        y_test_encoded = label_encoder.transform(y_test)
        
        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
        report = classification_report(y_test_encoded, y_pred_encoded, 
                                     target_names=label_encoder.classes_, 
                                     output_dict=True)
        conf_matrix = confusion_matrix(y_test_encoded, y_pred_encoded)
        
        evaluation_results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'class_names': label_encoder.classes_.tolist()
        }
        
        return render_template('views/lda/evaluate.html', 
                             results=evaluation_results,
                             stats=get_lda_statistics())
                             
    except Exception as e:
        flash(f'Error evaluasi: {str(e)}', 'error')
        return redirect(url_for('lda.index'))

@lda_bp.route('/detail/<int:id>')
def detail(id):
    """Detail hasil klasifikasi"""
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    hasil = db.session.query(
        HasilKlasifikasi, DataCitra, DatasetGambar, KategoriPenyakit
    ).join(
        DataCitra, HasilKlasifikasi.id_data_citra == DataCitra.id_data_citra
    ).join(
        DatasetGambar, DataCitra.id_gambar == DatasetGambar.id_gambar
    ).join(
        KategoriPenyakit, HasilKlasifikasi.id_kategori == KategoriPenyakit.id_kategori
    ).filter(HasilKlasifikasi.id_hasil == id).first()
    
    if not hasil:
        flash('Data tidak ditemukan', 'error')
        return redirect(url_for('lda.index'))
    
    hasil_data, citra_data, dataset_data, kategori_data = hasil
    dataset_data.fixed_image_path = fix_image_path(dataset_data.path_file)
    
    return render_template('views/lda/detail.html', hasil=hasil)

@lda_bp.route('/hapus/<int:id>')
def hapus(id):
    """Hapus hasil klasifikasi"""
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    hasil = HasilKlasifikasi.query.get_or_404(id)
    db.session.delete(hasil)
    db.session.commit()
    
    flash('Data berhasil dihapus', 'success')
    return redirect(url_for('lda.index'))

@lda_bp.route('/reset', methods=['POST'])
def reset():
    """Reset semua hasil dan model"""
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    deleted_count = HasilKlasifikasi.query.count()
    HasilKlasifikasi.query.delete()
    db.session.commit()
    
    model_path = os.path.join(MODEL_DIR, 'lda_model.pkl')
    if os.path.exists(model_path):
        os.remove(model_path)
    
    flash(f'Berhasil reset {deleted_count} hasil dan model', 'success')
    return redirect(url_for('lda.index'))

@lda_bp.route('/export_csv')
def export_csv():
    """Export hasil ke CSV"""
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    hasil_list = db.session.query(
        HasilKlasifikasi, DataCitra, DatasetGambar, KategoriPenyakit
    ).join(
        DataCitra, HasilKlasifikasi.id_data_citra == DataCitra.id_data_citra
    ).join(
        DatasetGambar, DataCitra.id_gambar == DatasetGambar.id_gambar
    ).join(
        KategoriPenyakit, HasilKlasifikasi.id_kategori == KategoriPenyakit.id_kategori
    ).all()
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow([
        'ID', 'Nama File', 'Kategori', 'Skor', 'Tanggal'
    ])
    
    for hasil, data_citra, dataset, kategori in hasil_list:
        writer.writerow([
            hasil.id_hasil,
            dataset.nama_file,
            kategori.nama_kategori,
            hasil.skor_lda,
            hasil.tanggal_klasifikasi.strftime('%Y-%m-%d %H:%M:%S')
        ])
    
    output.seek(0)
    
    filename = f'hasil_lda_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    
    response = make_response(output.getvalue())
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = f'attachment; filename={filename}'
    
    return response

@lda_bp.route('/api/stats')
def api_stats():
    """API statistik"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    stats = get_lda_statistics()
    return jsonify(stats)

@lda_bp.route('/test')
def test():
    """Test endpoint untuk debugging"""
    try:
        train_count = db.session.query(SplitData).filter_by(jenis_split='train').count()
        test_count = db.session.query(SplitData).filter_by(jenis_split='test').count()
        hasil_count = HasilKlasifikasi.query.count()
        model_exists = os.path.exists(os.path.join(MODEL_DIR, 'lda_model.pkl'))

        X_train, y_train, labels = prepare_training_data()
        train_ready = X_train is not None
        
        return jsonify({
            'train_count': train_count,
            'test_count': test_count,
            'hasil_count': hasil_count,
            'model_exists': model_exists,
            'train_ready': train_ready,
            'message': 'LDA test successful'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'LDA test failed'
        })