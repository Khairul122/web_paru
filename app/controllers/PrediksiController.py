from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
from app.extension import db
from app.models.AuthModel import User
from app.models.PrediksiModel import PrediksiGambar

prediksi_bp = Blueprint('prediksi', __name__)

UPLOAD_FOLDER = 'static/uploads/prediksi'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models', 'saved_models')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_upload_folder():
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (256, 256))
        normalized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized.astype(np.uint8)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def extract_glcm_features(image):
    try:
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        glcm = graycomatrix(image, distances=distances, angles=angles, 
                          levels=256, symmetric=True, normed=True)
        
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        asm = graycoprops(glcm, 'ASM')[0, 0]
        
        return {
            'contrast': float(contrast),
            'dissimilarity': float(dissimilarity),
            'homogeneity': float(homogeneity),
            'energy': float(energy),
            'correlation': float(correlation),
            'asm': float(asm)
        }
    except Exception as e:
        print(f"Error extracting GLCM features: {e}")
        return None

def predict_with_lda(features):
    try:
        model_path = os.path.join(MODEL_DIR, 'lda_model.pkl')
        
        if not os.path.exists(model_path):
            return None, 0.0
        
        model_data = joblib.load(model_path)
        lda_model = model_data['lda_model']
        scaler = model_data['scaler']
        label_encoder = model_data['label_encoder']
        
        feature_array = np.array([[
            features['contrast'],
            features['dissimilarity'],
            features['homogeneity'],
            features['energy'],
            features['correlation'],
            features['asm']
        ]])
        
        feature_scaled = scaler.transform(feature_array)
        prediction = lda_model.predict(feature_scaled)
        probabilities = lda_model.predict_proba(feature_scaled)
        
        predicted_class = label_encoder.inverse_transform(prediction)[0]
        confidence = np.max(probabilities)
        
        return predicted_class, float(confidence)
    except Exception as e:
        print(f"Error predicting with LDA: {e}")
        return None, 0.0

@prediksi_bp.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    page = request.args.get('page', 1, type=int)
    per_page = 10
    
    user_id = session['user_id']
    
    prediksi_list = PrediksiGambar.query.filter_by(id_user=user_id)\
                                       .order_by(PrediksiGambar.tanggal_prediksi.desc())\
                                       .paginate(page=page, per_page=per_page, error_out=False)
    
    return render_template('views/prediksi/index.html', prediksi_list=prediksi_list)

@prediksi_bp.route('/upload')
def upload():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    return render_template('views/prediksi/upload.html')

@prediksi_bp.route('/process', methods=['POST'])
def process():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    if 'file' not in request.files:
        flash('Tidak ada file yang dipilih', 'error')
        return redirect(url_for('prediksi.upload'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('Tidak ada file yang dipilih', 'error')
        return redirect(url_for('prediksi.upload'))
    
    if not allowed_file(file.filename):
        flash('Format file tidak didukung. Gunakan PNG, JPG, JPEG, BMP, atau TIFF', 'error')
        return redirect(url_for('prediksi.upload'))
    
    try:
        create_upload_folder()
        
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        file.save(file_path)
        file_size = os.path.getsize(file_path)
        
        preprocessed_image = preprocess_image(file_path)
        if preprocessed_image is None:
            flash('Gagal memproses gambar', 'error')
            return redirect(url_for('prediksi.upload'))
        
        glcm_features = extract_glcm_features(preprocessed_image)
        if glcm_features is None:
            flash('Gagal mengekstrak fitur GLCM', 'error')
            return redirect(url_for('prediksi.upload'))
        
        predicted_class, confidence = predict_with_lda(glcm_features)
        if predicted_class is None:
            flash('Model LDA belum tersedia. Lakukan training terlebih dahulu', 'error')
            return redirect(url_for('prediksi.upload'))
        
        prediksi = PrediksiGambar(
            id_user=session['user_id'],
            nama_file=file.filename,
            path_file=file_path,
            ukuran_file=file_size,
            contrast=glcm_features['contrast'],
            dissimilarity=glcm_features['dissimilarity'],
            homogeneity=glcm_features['homogeneity'],
            energy=glcm_features['energy'],
            correlation=glcm_features['correlation'],
            asm=glcm_features['asm'],
            prediksi_kategori=predicted_class,
            confidence_score=confidence
        )
        
        db.session.add(prediksi)
        db.session.commit()
        
        flash(f'Prediksi berhasil! Hasil: {predicted_class} (Confidence: {confidence:.2%})', 'success')
        return redirect(url_for('prediksi.detail', id=prediksi.id_prediksi))
        
    except Exception as e:
        flash(f'Terjadi kesalahan: {str(e)}', 'error')
        return redirect(url_for('prediksi.upload'))

@prediksi_bp.route('/detail/<int:id>')
def detail(id):
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    prediksi = PrediksiGambar.query.filter_by(id_prediksi=id, id_user=session['user_id']).first()
    
    if not prediksi:
        flash('Data prediksi tidak ditemukan', 'error')
        return redirect(url_for('prediksi.index'))
    
    return render_template('views/prediksi/detail.html', prediksi=prediksi)

@prediksi_bp.route('/hapus/<int:id>')
def hapus(id):
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    prediksi = PrediksiGambar.query.filter_by(id_prediksi=id, id_user=session['user_id']).first()
    
    if not prediksi:
        flash('Data prediksi tidak ditemukan', 'error')
        return redirect(url_for('prediksi.index'))
    
    try:
        if os.path.exists(prediksi.path_file):
            os.remove(prediksi.path_file)
        
        db.session.delete(prediksi)
        db.session.commit()
        
        flash('Data prediksi berhasil dihapus', 'success')
    except Exception as e:
        flash(f'Gagal menghapus data: {str(e)}', 'error')
    
    return redirect(url_for('prediksi.index'))

@prediksi_bp.route('/api/stats')
def api_stats():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        user_id = session['user_id']
        total_prediksi = PrediksiGambar.query.filter_by(id_user=user_id).count()
        
        recent_prediksi = PrediksiGambar.query.filter_by(id_user=user_id)\
                                             .order_by(PrediksiGambar.tanggal_prediksi.desc())\
                                             .limit(5).all()
        
        stats = {
            'total_prediksi': total_prediksi,
            'recent_count': len(recent_prediksi)
        }
        
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'error': str(e)}), 500