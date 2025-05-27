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
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(resized)
        print("CLAHE enhancement applied")
        
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        print("Sharpening filter applied")
        
        denoised = cv2.medianBlur(sharpened, 3)
        print("Median blur denoising applied")
        
        normalized = cv2.normalize(denoised, None, 0, 63, cv2.NORM_MINMAX)
        print(f"Normalized to range 0-63. Min: {np.min(normalized)}, Max: {np.max(normalized)}")
        
        result = normalized.astype(np.uint8)
        print(f"Final result shape: {result.shape}, dtype: {result.dtype}, range: {np.min(result)}-{np.max(result)}")
        
        return result
    
    except Exception as e:
        print(f"Error in load_and_preprocess_image: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def calculate_glcm_features(image):
    try:
        print("Starting enhanced GLCM calculation...")
        
        max_val = np.max(image)
        min_val = np.min(image)
        print(f"Image value range: {min_val} to {max_val}")
        
        if max_val >= 64:
            print("Rescaling image to fit 64 levels...")
            image = ((image - min_val) / (max_val - min_val) * 63).astype(np.uint8)
            print(f"Rescaled image range: {np.min(image)} to {np.max(image)}")
        
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        levels = 64
        
        print(f"Enhanced GLCM parameters - distances: {distances}, angles: {angles}, levels: {levels}")
        
        glcm = graycomatrix(
            image, 
            distances=distances, 
            angles=angles,
            levels=levels,
            symmetric=True,
            normed=True
        )
        
        print(f"GLCM matrix shape: {glcm.shape}")
        
        contrast_values = []
        dissimilarity_values = []
        homogeneity_values = []
        energy_values = []
        correlation_values = []
        asm_values = []
        
        for d_idx in range(len(distances)):
            for a_idx in range(len(angles)):
                glcm_slice = glcm[:, :, d_idx, a_idx]
                
                contrast_values.append(np.sum((np.arange(levels)[:, None] - np.arange(levels))**2 * glcm_slice))
                
                i, j = np.meshgrid(range(levels), range(levels), indexing='ij')
                dissimilarity_values.append(np.sum(np.abs(i - j) * glcm_slice))
                
                homogeneity_values.append(np.sum(glcm_slice / (1 + (i - j)**2)))
                
                energy_values.append(np.sum(glcm_slice**2))
                asm_values.append(np.sum(glcm_slice**2))
                
                mu_i = np.sum(i * glcm_slice)
                mu_j = np.sum(j * glcm_slice)
                std_i = np.sqrt(np.sum((i - mu_i)**2 * glcm_slice))
                std_j = np.sqrt(np.sum((j - mu_j)**2 * glcm_slice))
                
                if std_i > 0 and std_j > 0:
                    corr = np.sum((i - mu_i) * (j - mu_j) * glcm_slice) / (std_i * std_j)
                else:
                    corr = 0
                correlation_values.append(corr)
        
        contrast = np.mean(contrast_values)
        dissimilarity = np.mean(dissimilarity_values)
        homogeneity = np.mean(homogeneity_values)
        energy = np.mean(energy_values)
        correlation = np.mean(correlation_values)
        asm_value = np.mean(asm_values)
        
        features = {
            'contrast': float(contrast) if not np.isnan(contrast) and not np.isinf(contrast) else 0.0,
            'dissimilarity': float(dissimilarity) if not np.isnan(dissimilarity) and not np.isinf(dissimilarity) else 0.0,
            'homogeneity': float(homogeneity) if not np.isnan(homogeneity) and not np.isinf(homogeneity) else 0.0,
            'energy': float(energy) if not np.isnan(energy) and not np.isinf(energy) else 0.0,
            'correlation': float(correlation) if not np.isnan(correlation) and not np.isinf(correlation) else 0.0,
            'asm': float(asm_value) if not np.isnan(asm_value) and not np.isinf(asm_value) else 0.0
        }
        
        print(f"Enhanced GLCM features calculated: {features}")
        return features
    
    except Exception as e:
        print(f"Error calculating enhanced GLCM features: {str(e)}")
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
    try:
        import cv2
        import numpy as np
        from skimage.feature import graycomatrix, graycoprops
        from skimage import img_as_ubyte
        
        dummy_image = np.random.randint(0, 63, (256, 256), dtype=np.uint8)
        features = calculate_glcm_features(dummy_image)
        
        return {
            'status': 'success',
            'opencv_version': cv2.__version__,
            'numpy_version': np.__version__,
            'dummy_features': features,
            'message': 'All enhanced dependencies working'
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
    print("=== PROSES EKSTRAKSI ENHANCED DIMULAI ===")
    
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
                
                print("Loading and preprocessing image with enhanced method...")
                image = load_and_preprocess_image(image_path)
                if image is None:
                    error_count += 1
                    error_msg = f"Gagal memproses gambar {dataset.nama_file}"
                    error_messages.append(error_msg)
                    print(error_msg)
                    continue
                
                print("Image loaded successfully, calculating enhanced GLCM features...")
                features = calculate_glcm_features(image)
                print(f"Enhanced features calculated: {features}")
                
                if features and all(not np.isnan(v) and not np.isinf(v) and v >= 0 for v in features.values()):
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
                    print(f"SUCCESS: {dataset.nama_file} - Enhanced Features: {features}")
                else:
                    error_count += 1
                    error_msg = f"Gagal menghitung fitur enhanced GLCM untuk {dataset.nama_file} (nilai tidak valid)"
                    error_messages.append(error_msg)
                    print(error_msg)
            
            except Exception as e:
                error_count += 1
                error_msg = f"Error pada gambar ID {id_gambar}: {str(e)}"
                error_messages.append(error_msg)
                print(f"ERROR: {error_msg}")
                import traceback
                print(traceback.format_exc())
        
        print(f"\n=== HASIL EKSTRAKSI ENHANCED ===")
        print(f"Success: {success_count}, Error: {error_count}")
        
        if success_count > 0:
            db.session.commit()
            print("Database commit berhasil")
            flash(f'Berhasil mengekstraksi {success_count} gambar dengan metode enhanced', 'success')
        
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
    print("=== EKSTRAKSI SEMUA ENHANCED DIMULAI ===")
    
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
                
                if features and all(not np.isnan(v) and not np.isinf(v) and v >= 0 for v in features.values()):
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
                    error_messages.append(f"Gagal menghitung fitur enhanced GLCM untuk {dataset.nama_file}")
            
            except Exception as e:
                error_count += 1
                error_messages.append(f"Error pada {dataset.nama_file}: {str(e)}")
                print(f"Error processing {dataset.nama_file}: {str(e)}")
        
        if success_count > 0:
            db.session.commit()
        
        flash(f'Ekstraksi enhanced selesai. Berhasil: {success_count}, Gagal: {error_count}', 'success')
        
        if error_messages:
            for msg in error_messages[:5]:
                flash(msg, 'warning')
        
        return redirect(url_for('glcm.index'))
    
    except Exception as e:
        db.session.rollback()
        print(f"Error ekstraksi semua enhanced: {str(e)}")
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
        response.headers['Content-Disposition'] = f'attachment; filename=enhanced_glcm_features_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        return response
    
    except Exception as e:
        flash(f'Gagal export CSV: {str(e)}', 'error')
        return redirect(url_for('glcm.index'))