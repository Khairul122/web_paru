from flask import Blueprint, render_template, request, redirect, url_for, session
from app.extension import db
from app.models.KategoriModel import KategoriPenyakit
from app.models.AuthModel import User

kategori_bp = Blueprint('kategori', __name__)

@kategori_bp.route('/kategori')
def index():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))

    kategori_list = KategoriPenyakit.query.all()
    return render_template('views/kategori/index.html', kategori_list=kategori_list)

@kategori_bp.route('/kategori/tambah', methods=['GET', 'POST'])
def tambah():
    if request.method == 'POST':
        try:
            nama_kategori = request.form['nama_kategori']
            deskripsi = request.form['deskripsi']
            kategori = KategoriPenyakit(nama_kategori=nama_kategori, deskripsi=deskripsi)
            db.session.add(kategori)
            db.session.commit()
            return render_template('redirect.html',
                                   message='Kategori berhasil ditambahkan.',
                                   redirect_url=url_for('kategori.index'))
        except:
            db.session.rollback()
            return render_template('redirect.html',
                                   message='Gagal menambahkan kategori.',
                                   redirect_url=url_for('kategori.index'))
    return render_template('views/kategori/form.html', action='tambah')

@kategori_bp.route('/kategori/edit/<int:id>', methods=['GET', 'POST'])
def edit(id):
    kategori = KategoriPenyakit.query.get_or_404(id)
    if request.method == 'POST':
        try:
            kategori.nama_kategori = request.form['nama_kategori']
            kategori.deskripsi = request.form['deskripsi']
            db.session.commit()
            return render_template('redirect.html',
                                   message='Kategori berhasil diperbarui.',
                                   redirect_url=url_for('kategori.index'))
        except:
            db.session.rollback()
            return render_template('redirect.html',
                                   message='Gagal memperbarui kategori.',
                                   redirect_url=url_for('kategori.index'))
    return render_template('views/kategori/form.html', action='edit', kategori=kategori)

@kategori_bp.route('/kategori/hapus/<int:id>')
def hapus(id):
    kategori = KategoriPenyakit.query.get_or_404(id)
    try:
        db.session.delete(kategori)
        db.session.commit()
        return render_template('redirect.html',
                               message='Kategori berhasil dihapus.',
                               redirect_url=url_for('kategori.index'))
    except:
        db.session.rollback()
        return render_template('redirect.html',
                               message='Gagal menghapus kategori.',
                               redirect_url=url_for('kategori.index'))
