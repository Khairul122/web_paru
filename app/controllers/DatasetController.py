from flask import Blueprint, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from app.extension import db
from app.models.DatasetModel import DatasetGambar
from app.models.KategoriModel import KategoriPenyakit
import os

from datetime import datetime

dataset_bp = Blueprint('dataset', __name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, '..', 'static', 'uploads', 'dataset')

@dataset_bp.route('/dataset')
def index():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))

    dataset_list = DatasetGambar.query.all()
    return render_template('views/dataset/index.html', dataset_list=dataset_list)

@dataset_bp.route('/dataset/tambah-folder', methods=['GET', 'POST'])
def tambah_folder():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))

    kategori_list = KategoriPenyakit.query.all()

    if request.method == 'POST':
        id_kategori = int(request.form['id_kategori'])
        files = request.files.getlist('folder[]')

        kategori = KategoriPenyakit.query.get(id_kategori)
        prefix = kategori.nama_kategori.lower().replace(' ', '_')

        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        kategori_folder = os.path.join(UPLOAD_FOLDER, prefix)
        os.makedirs(kategori_folder, exist_ok=True)

        index_awal = DatasetGambar.query.filter_by(id_kategori=id_kategori).count() + 1

        for i, file in enumerate(files):
            if file and file.filename:
                ext = os.path.splitext(file.filename)[1]
                new_filename = f"{prefix}_{index_awal + i}{ext}"
                save_path = os.path.join(kategori_folder, new_filename)

                file.save(save_path)

                relative_path = os.path.join('static', 'uploads', 'dataset', prefix, new_filename)

                gambar = DatasetGambar(
                    id_kategori=id_kategori,
                    nama_file=new_filename,
                    path_file=relative_path
                )
                db.session.add(gambar)

        db.session.commit()

        return render_template('redirect.html',
                               message='Semua gambar berhasil diunggah dan disimpan.',
                               redirect_url=url_for('dataset.index'))

    return render_template('views/dataset/form_folder.html', kategori_list=kategori_list)

@dataset_bp.route('/dataset/edit/<int:id>', methods=['GET', 'POST'])
def edit(id):
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))

    dataset = DatasetGambar.query.get_or_404(id)
    kategori_list = KategoriPenyakit.query.all()

    if request.method == 'POST':
        old_kategori_id = dataset.id_kategori
        new_kategori_id = int(request.form['id_kategori'])
        
        if old_kategori_id != new_kategori_id:
            old_kategori = KategoriPenyakit.query.get(old_kategori_id)
            new_kategori = KategoriPenyakit.query.get(new_kategori_id)
            
            old_prefix = old_kategori.nama_kategori.lower().replace(' ', '_')
            new_prefix = new_kategori.nama_kategori.lower().replace(' ', '_')
            
            old_file_path = os.path.join(BASE_DIR, '..', dataset.path_file)
            
            new_kategori_folder = os.path.join(UPLOAD_FOLDER, new_prefix)
            os.makedirs(new_kategori_folder, exist_ok=True)
            
            ext = os.path.splitext(dataset.nama_file)[1]
            index_baru = DatasetGambar.query.filter_by(id_kategori=new_kategori_id).count() + 1
            new_filename = f"{new_prefix}_{index_baru}{ext}"
            new_file_path = os.path.join(new_kategori_folder, new_filename)
            
            if os.path.exists(old_file_path):
                os.rename(old_file_path, new_file_path)
            
            dataset.id_kategori = new_kategori_id
            dataset.nama_file = new_filename
            dataset.path_file = os.path.join('static', 'uploads', 'dataset', new_prefix, new_filename)
        else:
            dataset.id_kategori = new_kategori_id
        
        db.session.commit()
        return render_template('redirect.html',
                               message='Data berhasil diperbarui.',
                               redirect_url=url_for('dataset.index'))

    return render_template('views/dataset/form_edit.html', dataset=dataset, kategori_list=kategori_list)

@dataset_bp.route('/dataset/hapus/<int:id>')
def hapus(id):
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))

    dataset = DatasetGambar.query.get_or_404(id)
    try:
        # Hapus file fisik jika ada
        full_path = os.path.join(BASE_DIR, '..', dataset.path_file)
        if os.path.exists(full_path):
            os.remove(full_path)
            
            # Cek apakah folder kategori kosong, jika ya hapus folder
            kategori = KategoriPenyakit.query.get(dataset.id_kategori)
            kategori_prefix = kategori.nama_kategori.lower().replace(' ', '_')
            kategori_folder = os.path.join(UPLOAD_FOLDER, kategori_prefix)
            
            if os.path.exists(kategori_folder) and not os.listdir(kategori_folder):
                os.rmdir(kategori_folder)

        db.session.delete(dataset)
        db.session.commit()

        return render_template('redirect.html',
                               message='Data berhasil dihapus.',
                               redirect_url=url_for('dataset.index'))
    except Exception as e:
        db.session.rollback()
        return render_template('redirect.html',
                               message='Gagal menghapus data.',
                               redirect_url=url_for('dataset.index'))