from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify, make_response
from app.extension import db
from app.models.DatasetModel import DatasetGambar
from app.models.DataCitraModel import DataCitra
import os
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
from datetime import datetime
import csv
import io

glcm_bp = Blueprint('glcm', __name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def load_and_preprocess_image(image_path):
    """
    Load image dan preprocessing untuk GLCM
    """
    try:
        print(f"Trying to load image: {image_path}")
        
        image_path = os.path.normpath(image_path)
        print(f"Normalized path: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"File tidak ditemukan: {image_path}")
            return None
            
        image = cv2.imread(image_path)
        if image is None:
            print(f"cv2.imread gagal membaca: {image_path}")
            return None
        
        print(f"Image loaded successfully. Shape: {image.shape}")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(f"Converted to grayscale. Shape: {gray.shape}")
        
        resized = cv2.resize(gray, (256, 256))
        print(f"Resized to 256x256. Shape: {resized.shape}")
        
        normalized = cv2.equalizeHist(resized)
        print("Histogram equalization applied")
        
        result = img_as_ubyte(normalized)
        print(f"Final result shape: {result.shape}, dtype: {result.dtype}")
        
        return result
    
    except Exception as e:
        print(f"Error in load_and_preprocess_image: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def calculate_glcm_features(image):
    """
    Menghitung fitur GLCM dari gambar
    """
    try:
        print("Starting GLCM calculation...")
        
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        levels = 256
        
        print(f"GLCM parameters - distances: {distances}, angles: {angles}, levels: {levels}")
        
        glcm = graycomatrix(
            image, 
            distances=distances, 
            angles=angles,
            levels=levels,
            symmetric=True,
            normed=True
        )
        
        print(f"GLCM matrix shape: {glcm.shape}")
        
        contrast = np.mean(graycoprops(glcm, 'contrast'))
        dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
        homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
        energy = np.mean(graycoprops(glcm, 'energy'))
        correlation = np.mean(graycoprops(glcm, 'correlation'))
        asm_value = np.mean(graycoprops(glcm, 'ASM'))
        
        features = {
            'contrast': float(contrast),
            'dissimilarity': float(dissimilarity),
            'homogeneity': float(homogeneity),
            'energy': float(energy),
            'correlation': float(correlation),
            'asm': float(asm_value)
        }
        
        print(f"GLCM features calculated: {features}")
        return features
    
    except Exception as e:
        print(f"Error calculating GLCM features: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

@glcm_bp.route('/glcm')
def index():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))

    page = request.args.get('page', 1, type=int)
    per_page = 10

    data_citra_list = db.session.query(DataCitra, DatasetGambar).join(
        DatasetGambar, DataCitra.id_gambar == DatasetGambar.id_gambar
    ).paginate(page=page, per_page=per_page)

    return render_template('views/glcm/index.html', data_citra_list=data_citra_list)


@glcm_bp.route('/glcm/ekstraksi')
def ekstraksi():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    dataset_list = DatasetGambar.query.filter(
        ~DatasetGambar.id_gambar.in_(
            db.session.query(DataCitra.id_gambar)
        )
    ).all()
    
    return render_template('views/glcm/ekstraksi.html', dataset_list=dataset_list)

@glcm_bp.route('/glcm/test')
def test():
    """Route untuk test dependencies dan fungsi GLCM"""
    try:
        import cv2
        import numpy as np
        from skimage.feature import graycomatrix, graycoprops
        from skimage import img_as_ubyte
        
        dummy_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        features = calculate_glcm_features(dummy_image)
        
        return {
            'status': 'success',
            'opencv_version': cv2.__version__,
            'numpy_version': np.__version__,
            'dummy_features': features,
            'message': 'All dependencies working'
        }
    except Exception as e:
        import traceback
        return {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }

@glcm_bp.route('/glcm/proses-ekstraksi', methods=['POST'])
def proses_ekstraksi():
    print("=== PROSES EKSTRAKSI DIMULAI ===")
    
    if 'user_id' not in session:
        print("User tidak login, redirect ke login")
        flash('Silakan login terlebih dahulu', 'error')
        return redirect(url_for('auth.login'))
    
    print(f"User ID: {session['user_id']}")
    
    try:
        selected_images = request.form.getlist('selected_images')
        print(f"Selected images: {selected_images}")
        
        if not selected_images:
            print("Tidak ada gambar yang dipilih")
            flash('Pilih minimal satu gambar untuk diekstraksi', 'error')
            return redirect(url_for('glcm.ekstraksi'))
        
        success_count = 0
        error_count = 0
        error_messages = []
        
        for id_gambar in selected_images:
            print(f"\n--- Processing image ID: {id_gambar} ---")
            
            try:
                dataset = DatasetGambar.query.get(id_gambar)
                if not dataset:
                    error_count += 1
                    error_msg = f"Dataset dengan ID {id_gambar} tidak ditemukan"
                    error_messages.append(error_msg)
                    print(error_msg)
                    continue
                
                print(f"Dataset found: {dataset.nama_file}")
                
                existing_data = DataCitra.query.filter_by(id_gambar=id_gambar).first()
                if existing_data:
                    error_count += 1
                    error_msg = f"Gambar {dataset.nama_file} sudah diekstraksi"
                    error_messages.append(error_msg)
                    print(error_msg)
                    continue
                
                image_path = os.path.join(BASE_DIR, '..', dataset.path_file.replace('\\', os.sep).replace('/', os.sep))
                image_path = os.path.normpath(image_path)
                print(f"Image path: {image_path}")
                print(f"File exists: {os.path.exists(image_path)}")
                
                if not os.path.exists(image_path):
                    error_count += 1
                    error_msg = f"File {dataset.nama_file} tidak ditemukan di path: {image_path}"
                    error_messages.append(error_msg)
                    print(error_msg)
                    continue
                
                print("Loading and preprocessing image...")
                image = load_and_preprocess_image(image_path)
                if image is None:
                    error_count += 1
                    error_msg = f"Gagal memproses gambar {dataset.nama_file}"
                    error_messages.append(error_msg)
                    print(error_msg)
                    continue
                
                print("Image loaded successfully, calculating GLCM features...")
                features = calculate_glcm_features(image)
                print(f"Features calculated: {features}")
                
                if features and all(not np.isnan(v) and not np.isinf(v) for v in features.values()):
                    data_citra = DataCitra(
                        id_gambar=int(id_gambar),
                        contrast=features['contrast'],
                        dissimilarity=features['dissimilarity'],
                        homogeneity=features['homogeneity'],
                        energy=features['energy'],
                        correlation=features['correlation'],
                        asm=features['asm'],
                        uploaded_by=session['user_id']
                    )
                    
                    db.session.add(data_citra)
                    success_count += 1
                    print(f"SUCCESS: {dataset.nama_file} - Features: {features}")
                else:
                    error_count += 1
                    error_msg = f"Gagal menghitung fitur GLCM untuk {dataset.nama_file} (nilai tidak valid)"
                    error_messages.append(error_msg)
                    print(error_msg)
            
            except Exception as e:
                error_count += 1
                error_msg = f"Error pada gambar ID {id_gambar}: {str(e)}"
                error_messages.append(error_msg)
                print(f"ERROR: {error_msg}")
                import traceback
                print(traceback.format_exc())
        
        print(f"\n=== HASIL EKSTRAKSI ===")
        print(f"Success: {success_count}, Error: {error_count}")
        
        if success_count > 0:
            db.session.commit()
            print("Database commit berhasil")
            flash(f'Berhasil mengekstraksi {success_count} gambar', 'success')
        
        if error_count > 0:
            flash(f'Gagal mengekstraksi {error_count} gambar', 'warning')
            for msg in error_messages[:3]:
                flash(msg, 'warning')
        
        if success_count == 0 and error_count == 0:
            flash('Tidak ada gambar yang diproses', 'info')
        
        print("Redirecting to GLCM index...")
        return redirect(url_for('glcm.index'))
    
    except Exception as e:
        db.session.rollback()
        error_msg = f"Error general: {str(e)}"
        print(f"GENERAL ERROR: {error_msg}")
        import traceback
        print(traceback.format_exc())
        flash(f'Terjadi error: {str(e)}', 'error')
        return redirect(url_for('glcm.ekstraksi'))

@glcm_bp.route('/glcm/ekstraksi-semua', methods=['POST'])
def ekstraksi_semua():
    print("=== EKSTRAKSI SEMUA DIMULAI ===")
    
    if 'user_id' not in session:
        flash('Silakan login terlebih dahulu', 'error')
        return redirect(url_for('auth.login'))
    
    try:
        dataset_list = DatasetGambar.query.filter(
            ~DatasetGambar.id_gambar.in_(
                db.session.query(DataCitra.id_gambar)
            )
        ).all()
        
        if not dataset_list:
            flash('Tidak ada gambar yang belum diekstraksi', 'info')
            return redirect(url_for('glcm.index'))
        
        success_count = 0
        error_count = 0
        error_messages = []
        
        for dataset in dataset_list:
            print(f"\n--- Processing: {dataset.nama_file} (ID: {dataset.id_gambar}) ---")
            
            try:
                image_path = os.path.join(BASE_DIR, '..', dataset.path_file.replace('\\', os.sep).replace('/', os.sep))
                image_path = os.path.normpath(image_path)
                
                if not os.path.exists(image_path):
                    print(f"File tidak ditemukan: {image_path}")
                    error_count += 1
                    error_messages.append(f"File {dataset.nama_file} tidak ditemukan")
                    continue
                
                image = load_and_preprocess_image(image_path)
                if image is None:
                    error_count += 1
                    error_messages.append(f"Gagal memproses gambar {dataset.nama_file}")
                    continue
                
                features = calculate_glcm_features(image)
                
                if features and all(not np.isnan(v) and not np.isinf(v) for v in features.values()):
                    data_citra = DataCitra(
                        id_gambar=dataset.id_gambar,
                        contrast=features['contrast'],
                        dissimilarity=features['dissimilarity'],
                        homogeneity=features['homogeneity'],
                        energy=features['energy'],
                        correlation=features['correlation'],
                        asm=features['asm'],
                        uploaded_by=session['user_id']
                    )
                    
                    db.session.add(data_citra)
                    success_count += 1
                    print(f"SUCCESS: {dataset.nama_file}")
                else:
                    error_count += 1
                    error_messages.append(f"Gagal menghitung fitur GLCM untuk {dataset.nama_file}")
            
            except Exception as e:
                error_count += 1
                error_messages.append(f"Error pada {dataset.nama_file}: {str(e)}")
                print(f"Error processing {dataset.nama_file}: {str(e)}")
        
        if success_count > 0:
            db.session.commit()
        
        flash(f'Ekstraksi selesai. Berhasil: {success_count}, Gagal: {error_count}', 'success')
        
        if error_messages:
            for msg in error_messages[:5]:
                flash(msg, 'warning')
        
        return redirect(url_for('glcm.index'))
    
    except Exception as e:
        db.session.rollback()
        print(f"Error ekstraksi semua: {str(e)}")
        flash(f'Terjadi error: {str(e)}', 'error')
        return redirect(url_for('glcm.ekstraksi'))

@glcm_bp.route('/glcm/detail/<int:id>')
def detail(id):
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    data_citra = db.session.query(DataCitra, DatasetGambar).join(
        DatasetGambar, DataCitra.id_gambar == DatasetGambar.id_gambar
    ).filter(DataCitra.id_data_citra == id).first()
    
    if not data_citra:
        flash('Data tidak ditemukan', 'error')
        return redirect(url_for('glcm.index'))
    
    return render_template('views/glcm/detail.html', data_citra=data_citra)

@glcm_bp.route('/glcm/hapus/<int:id>')
def hapus(id):
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    try:
        data_citra = DataCitra.query.get_or_404(id)
        db.session.delete(data_citra)
        db.session.commit()
        
        flash('Data fitur GLCM berhasil dihapus', 'success')
        return redirect(url_for('glcm.index'))
    
    except Exception as e:
        db.session.rollback()
        flash(f'Gagal menghapus data: {str(e)}', 'error')
        return redirect(url_for('glcm.index'))

@glcm_bp.route('/glcm/export_csv')
def export_csv():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    try:
        data_citra_list = db.session.query(DataCitra, DatasetGambar).join(
            DatasetGambar, DataCitra.id_gambar == DatasetGambar.id_gambar
        ).all()
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        writer.writerow([
            'ID', 'Nama File', 'Kategori', 'Contrast', 'Dissimilarity', 
            'Homogeneity', 'Energy', 'Correlation', 'ASM', 'Tanggal Upload'
        ])
        
        for data_citra, dataset in data_citra_list:
            writer.writerow([
                data_citra.id_data_citra,
                dataset.nama_file,
                dataset.kategori.nama_kategori if dataset.kategori else '-',
                data_citra.contrast,
                data_citra.dissimilarity,
                data_citra.homogeneity,
                data_citra.energy,
                data_citra.correlation,
                data_citra.asm,
                data_citra.tanggal_upload.strftime('%Y-%m-%d %H:%M:%S')
            ])
        
        output.seek(0)
        
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename=glcm_features_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        return response
    
    except Exception as e:
        flash(f'Gagal export CSV: {str(e)}', 'error')
        return redirect(url_for('glcm.index'))