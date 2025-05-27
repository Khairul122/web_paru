# LDAModel.py
from app.extension import db
from datetime import datetime

class HasilKlasifikasi(db.Model):
    __tablename__ = 'hasil_klasifikasi'
    
    id_hasil = db.Column(db.Integer, primary_key=True)
    id_data_citra = db.Column(db.Integer, db.ForeignKey('data_citra.id_data_citra'), nullable=False)
    id_kategori = db.Column(db.Integer, db.ForeignKey('kategori_penyakit.id_kategori'), nullable=False)
    skor_lda = db.Column(db.Float)
    tanggal_klasifikasi = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<HasilKlasifikasi {self.id_hasil}>'
    
    def to_dict(self):
        return {
            'id_hasil': self.id_hasil,
            'id_data_citra': self.id_data_citra,
            'id_kategori': self.id_kategori,
            'skor_lda': self.skor_lda,
            'tanggal_klasifikasi': self.tanggal_klasifikasi.isoformat() if self.tanggal_klasifikasi else None
        }
