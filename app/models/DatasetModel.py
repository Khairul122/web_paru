from app.extension import db
from sqlalchemy.orm import relationship

class DatasetGambar(db.Model):
    __tablename__ = 'dataset_gambar'

    id_gambar = db.Column(db.Integer, primary_key=True)
    id_kategori = db.Column(db.Integer, db.ForeignKey('kategori_penyakit.id_kategori', ondelete='CASCADE'), nullable=False)
    nama_file = db.Column(db.String(255), nullable=False)
    path_file = db.Column(db.String(255), nullable=False)
    tanggal_upload = db.Column(db.DateTime, server_default=db.func.current_timestamp())

    kategori = relationship('KategoriPenyakit', backref='gambar')
