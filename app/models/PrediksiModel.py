from app import db
from datetime import datetime

class PrediksiGambar(db.Model):
    __tablename__ = 'prediksi_gambar'
    
    id_prediksi = db.Column(db.Integer, primary_key=True)
    id_user = db.Column(db.Integer, db.ForeignKey('users.id_user'), nullable=False)
    nama_file = db.Column(db.String(255), nullable=False)
    path_file = db.Column(db.String(500), nullable=False)
    ukuran_file = db.Column(db.Integer, nullable=True)
    contrast = db.Column(db.Float, nullable=True)
    dissimilarity = db.Column(db.Float, nullable=True)
    homogeneity = db.Column(db.Float, nullable=True)
    energy = db.Column(db.Float, nullable=True)
    correlation = db.Column(db.Float, nullable=True)
    asm = db.Column(db.Float, nullable=True)
    prediksi_kategori = db.Column(db.String(100), nullable=True)
    confidence_score = db.Column(db.Float, nullable=True)
    tanggal_prediksi = db.Column(db.DateTime, server_default=db.func.now())
    
    user = db.relationship('User', backref=db.backref('prediksi_gambar', lazy=True))
    
    def __repr__(self):
        return f'<PrediksiGambar {self.nama_file}>'