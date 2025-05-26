from app import db

class KategoriPenyakit(db.Model):
    __tablename__ = 'kategori_penyakit'

    id_kategori = db.Column(db.Integer, primary_key=True)
    nama_kategori = db.Column(db.String(100), nullable=False)
    deskripsi = db.Column(db.Text)
