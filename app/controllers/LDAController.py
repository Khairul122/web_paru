from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify, make_response
from app.extension import db
from app.models.DataCitraModel import DataCitra
from app.models.DatasetModel import DatasetGambar
from app.models.KategoriModel import KategoriPenyakit
from app.models.SplitDataModel import SplitData
from app.models.LDAModel import HasilKlasifikasi
from sqlalchemy import func, desc, asc
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from scipy import stats
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
    print(f"[LDA ULTRA] {message}")
    sys.stdout.flush()

def fix_image_path(path_file):
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

def create_smart_features(contrast, dissimilarity, homogeneity, energy, correlation, asm):
    eps = 1e-8
    
    contrast = max(contrast, eps)
    dissimilarity = max(dissimilarity, eps) 
    homogeneity = max(homogeneity, eps)
    energy = max(energy, eps)
    asm = max(asm, eps)
    
    features = []
    
    features.append(contrast)
    features.append(dissimilarity)
    features.append(homogeneity)
    features.append(energy)
    features.append(correlation)
    features.append(asm)
    
    features.append(contrast / energy)
    features.append(homogeneity / dissimilarity)
    features.append(energy / asm if asm > eps else 1.0)
    features.append(contrast * homogeneity)
    features.append(np.sqrt(energy * asm))
    features.append(abs(correlation) * energy)
    
    return features

def prepare_training_data():
    try:
        simple_log("=== MULAI PERSIAPAN DATA TRAINING ULTRA ===")

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
            contrast = float(row.contrast or 0)
            dissimilarity = float(row.dissimilarity or 0)
            homogeneity = float(row.homogeneity or 0)
            energy = float(row.energy or 0)
            correlation = float(row.correlation or 0)
            asm = float(row.asm or 0)
            
            feature_values = create_smart_features(
                contrast, dissimilarity, homogeneity, 
                energy, correlation, asm
            )
            
            features.append(feature_values)
            labels.append(row.nama_kategori)
        
        X = np.array(features)
        y = np.array(labels)
        
        simple_log(f"Initial features shape: {X.shape}")
        
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        simple_log(f"Mean per fitur: {means}")
        simple_log(f"Std dev per fitur: {stds}")

        z_scores = np.abs(stats.zscore(X, axis=0, nan_policy='omit'))
        outlier_mask = (z_scores < 2.5).all(axis=1)
        
        if np.sum(outlier_mask) == 0:
            simple_log("PERINGATAN: Semua data dianggap outlier. Menggunakan semua data tanpa filter.")
            outlier_mask = np.ones(len(X), dtype=bool)
        
        X = X[outlier_mask]
        y = y[outlier_mask]
        
        simple_log(f"After outlier removal: {X.shape}")
        
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        
        unique_labels = list(set(y))
        
        simple_log(f"Ultra Training data shape: {X.shape}")
        simple_log(f"Unique labels: {unique_labels}")
        simple_log(f"Label counts: {np.unique(y, return_counts=True)}")
        
        return X, y, unique_labels
        
    except Exception as e:
        simple_log(f"ERROR dalam prepare_training_data: {str(e)}")
        import traceback
        simple_log(traceback.format_exc())
        return None, None, None


def prepare_test_data():
    try:
        simple_log("=== MULAI PERSIAPAN DATA TESTING ULTRA ===")
        
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
            contrast = float(row.contrast or 0)
            dissimilarity = float(row.dissimilarity or 0)
            homogeneity = float(row.homogeneity or 0)
            energy = float(row.energy or 0)
            correlation = float(row.correlation or 0)
            asm = float(row.asm or 0)
            
            features.append([
                float(contrast), float(dissimilarity), float(homogeneity),
                float(energy), float(correlation), float(asm)
        ])

            features.append(feature_values)
            labels.append(row.nama_kategori)
            data_ids.append(row.id_data_citra)
        
        X = np.array(features)
        y = np.array(labels)
        
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        
        simple_log(f"Ultra Test data shape: {X.shape}")
        
        return X, y, data_ids, None
        
    except Exception as e:
        simple_log(f"ERROR dalam prepare_test_data: {str(e)}")
        return None, None, None, None

def get_lda_statistics():
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
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    stats = get_lda_statistics()
    return render_template('views/lda/train.html', stats=stats)

@lda_bp.route('/process_train', methods=['POST'])
def process_train():
    if 'user_id' not in session:
        flash('Silakan login terlebih dahulu', 'error')
        return redirect(url_for('auth.login'))
    
    try:
        simple_log("========================================")
        simple_log("=== MULAI PROSES TRAINING LDA ULTRA ===")
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
        
        simple_log("Data valid untuk training")
        
        simple_log("STEP 3: Ultra preprocessing pipeline...")
        
        robust_scaler = RobustScaler()
        X_robust = robust_scaler.fit_transform(X_train)
        
        standard_scaler = StandardScaler()
        X_scaled = standard_scaler.fit_transform(X_robust)
        
        simple_log(f"Data preprocessing selesai: {X_scaled.shape}")
        
        simple_log("STEP 4: PCA untuk dimensionality reduction...")
        pca = PCA(n_components=0.95, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        simple_log(f"PCA shape: {X_pca.shape}, explained variance: {pca.explained_variance_ratio_.sum():.4f}")
        
        simple_log("STEP 5: Feature selection...")
        n_features = min(8, X_pca.shape[1])
        selector = SelectKBest(score_func=f_classif, k=n_features)
        X_selected = selector.fit_transform(X_pca, y_train)
        simple_log(f"Feature selection selesai: {X_selected.shape}")
        
        simple_log("STEP 6: Encoding labels...")
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        simple_log(f"Labels encoded: {label_encoder.classes_}")

        simple_log("STEP 7: Ultra LDA training dengan ensemble...")
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        lda_models = []
        best_lda = None
        best_score = 0
        
        shrinkage_values = [None, 'auto', 0.1, 0.3, 0.5, 0.7]
        
        for shrinkage in shrinkage_values:
            try:
                if shrinkage is None:
                    lda = LinearDiscriminantAnalysis(solver='svd')
                else:
                    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=shrinkage)
                
                cv_scores = cross_val_score(lda, X_selected, y_train_encoded, cv=cv, scoring='accuracy')
                avg_score = np.mean(cv_scores)
                
                simple_log(f"LDA shrinkage {shrinkage}: CV = {avg_score:.4f} ± {np.std(cv_scores):.4f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_lda = lda
                    
            except Exception as e:
                simple_log(f"Error dengan shrinkage {shrinkage}: {str(e)}")
                continue
        
        if best_lda is None:
            best_lda = LinearDiscriminantAnalysis(solver='svd')
        
        best_lda.fit(X_selected, y_train_encoded)
        simple_log(f"Best LDA trained dengan CV score: {best_score:.4f}")
        
        simple_log("STEP 8: Training QDA sebagai backup...")
        try:
            qda = QuadraticDiscriminantAnalysis()
            qda_scores = cross_val_score(qda, X_selected, y_train_encoded, cv=cv, scoring='accuracy')
            qda_score = np.mean(qda_scores)
            simple_log(f"QDA CV score: {qda_score:.4f}")
            qda.fit(X_selected, y_train_encoded)
        except Exception as e:
            simple_log(f"QDA failed: {str(e)}")
            qda = None
            qda_score = 0

        simple_log("STEP 9: Final model selection...")
        if qda is not None and qda_score > best_score:
            final_model = qda
            final_score = qda_score
            model_type = 'QDA'
            simple_log("QDA selected as final model")
        else:
            final_model = best_lda
            final_score = best_score
            model_type = 'LDA'
            simple_log("LDA selected as final model")

        simple_log("STEP 10: Evaluasi model final...")
        y_train_pred = final_model.predict(X_selected)
        train_accuracy = accuracy_score(y_train_encoded, y_train_pred)
        simple_log(f"Training accuracy: {train_accuracy:.4f}")
        
        simple_log("STEP 11: Menyimpan ultra model...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        model_data = {
            'model': final_model,
            'model_type': model_type,
            'robust_scaler': robust_scaler,
            'standard_scaler': standard_scaler,
            'pca': pca,
            'selector': selector,
            'label_encoder': label_encoder,
            'feature_names': ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'asm'],
            'class_names': label_names,
            'training_accuracy': train_accuracy,
            'cv_accuracy': final_score,
            'pca_components': X_pca.shape[1],
            'selected_features': X_selected.shape[1],
            'trained_at': datetime.now(),
            'model_version': 'ultra_v4'
        }
        
        model_path = os.path.join(MODEL_DIR, 'lda_model.pkl')
        joblib.dump(model_data, model_path)
        simple_log(f"Ultra model disimpan di: {model_path}")
        
        simple_log("========================================")
        simple_log("=== ULTRA TRAINING BERHASIL SELESAI! ===")
        simple_log("========================================")
        
        flash(f'Ultra {model_type} Training berhasil! Model telah dilatih dengan akurasi: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)', 'success')
        return redirect(url_for('lda.train'))
        
    except Exception as e:
        simple_log(f"========================================")
        simple_log(f"=== ERROR DALAM ULTRA TRAINING: {str(e)} ===")
        simple_log("========================================")
        import traceback
        simple_log(traceback.format_exc())
        flash(f'Error dalam training: {str(e)}', 'error')
        return redirect(url_for('lda.train'))

@lda_bp.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    try:
        simple_log("=== MULAI ULTRA PREDIKSI ===")
        
        model_path = os.path.join(MODEL_DIR, 'lda_model.pkl')
        
        if not os.path.exists(model_path):
            simple_log("ERROR: Model tidak ditemukan!")
            flash('Model belum dilatih. Silakan lakukan training terlebih dahulu.', 'error')
            return redirect(url_for('lda.train'))
        
        simple_log("Loading ultra model...")
        model_data = joblib.load(model_path)
        model = model_data['model']
        model_type = model_data['model_type']
        robust_scaler = model_data['robust_scaler']
        standard_scaler = model_data['standard_scaler']
        pca = model_data['pca']
        selector = model_data['selector']
        label_encoder = model_data['label_encoder']
        
        X_test, y_test, data_ids, _ = prepare_test_data()
        
        if X_test is None:
            simple_log("ERROR: Tidak ada data test!")
            flash('Tidak ada data test. Silakan lakukan split data terlebih dahulu.', 'error')
            return redirect(url_for('lda.index'))
        
        simple_log("Melakukan ultra preprocessing dan prediksi...")
        X_test_robust = robust_scaler.transform(X_test)
        X_test_scaled = standard_scaler.transform(X_test_robust)
        X_test_pca = pca.transform(X_test_scaled)
        X_test_selected = selector.transform(X_test_pca)
        
        y_pred_encoded = model.predict(X_test_selected)
        y_pred_proba = model.predict_proba(X_test_selected)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        
        simple_log("Menyimpan hasil ultra prediksi...")
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
        
        simple_log(f"=== ULTRA PREDIKSI SELESAI ===")
        simple_log(f"Model: {model_type}, Berhasil: {success_count}, Akurasi: {test_accuracy:.4f}")
        
        flash(f'Ultra {model_type} Prediksi berhasil! {success_count} hasil disimpan dengan akurasi: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)', 'success')
        return redirect(url_for('lda.index'))
        
    except Exception as e:
        simple_log(f"ERROR dalam ultra prediksi: {str(e)}")
        flash(f'Error dalam prediksi: {str(e)}', 'error')
        return redirect(url_for('lda.index'))

@lda_bp.route('/evaluate')
def evaluate():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    try:
        model_path = os.path.join(MODEL_DIR, 'lda_model.pkl')
        
        if not os.path.exists(model_path):
            flash('Model belum dilatih', 'error')
            return redirect(url_for('lda.train'))
        
        model_data = joblib.load(model_path)
        model = model_data['model']
        model_type = model_data['model_type']
        robust_scaler = model_data['robust_scaler']
        standard_scaler = model_data['standard_scaler']
        pca = model_data['pca']
        selector = model_data['selector']
        label_encoder = model_data['label_encoder']
        
        X_test, y_test, data_ids, _ = prepare_test_data()
        
        if X_test is None:
            flash('Tidak ada data test', 'error')
            return redirect(url_for('lda.index'))
        
        X_test_robust = robust_scaler.transform(X_test)
        X_test_scaled = standard_scaler.transform(X_test_robust)
        X_test_pca = pca.transform(X_test_scaled)
        X_test_selected = selector.transform(X_test_pca)
        
        y_pred_encoded = model.predict(X_test_selected)
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
            'class_names': label_encoder.classes_.tolist(),
            'model_info': {
                'model_type': model_type,
                'training_accuracy': model_data.get('training_accuracy', 0),
                'cv_accuracy': model_data.get('cv_accuracy', 0),
                'pca_components': model_data.get('pca_components', 0),
                'selected_features': model_data.get('selected_features', 0),
                'model_version': model_data.get('model_version', 'unknown')
            }
        }
        
        return render_template('views/lda/evaluate.html', 
                             results=evaluation_results,
                             stats=get_lda_statistics())
                             
    except Exception as e:
        flash(f'Error evaluasi: {str(e)}', 'error')
        return redirect(url_for('lda.index'))

@lda_bp.route('/detail/<int:id>')
def detail(id):
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
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    hasil = HasilKlasifikasi.query.get_or_404(id)
    db.session.delete(hasil)
    db.session.commit()
    
    flash('Data berhasil dihapus', 'success')
    return redirect(url_for('lda.index'))

@lda_bp.route('/reset', methods=['POST'])
def reset():
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
    
    filename = f'hasil_lda_ultra_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    
    response = make_response(output.getvalue())
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = f'attachment; filename={filename}'
    
    return response

@lda_bp.route('/api/stats')
def api_stats():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    stats = get_lda_statistics()
    return jsonify(stats)

@lda_bp.route('/test')
def test():
    try:
        train_count = db.session.query(SplitData).filter_by(jenis_split='train').count()
        test_count = db.session.query(SplitData).filter_by(jenis_split='test').count()
        hasil_count = HasilKlasifikasi.query.count()
        model_exists = os.path.exists(os.path.join(MODEL_DIR, 'lda_model.pkl'))

        X_train, y_train, labels = prepare_training_data()
        train_ready = X_train is not None
        
        feature_info = {}
        if X_train is not None:
            feature_info = {
                'n_samples': X_train.shape[0],
                'n_features': X_train.shape[1], 
                'n_classes': len(labels) if labels else 0,
                'class_distribution': dict(zip(*np.unique(y_train, return_counts=True))) if y_train is not None else {}
            }
        
        return jsonify({
            'train_count': train_count,
            'test_count': test_count,
            'hasil_count': hasil_count,
            'model_exists': model_exists,
            'train_ready': train_ready,
            'feature_info': feature_info,
            'message': 'Ultra LDA test successful'
        })
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc(),
            'message': 'Ultra LDA test failed'
        })