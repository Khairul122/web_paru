from app.extension import db
from datetime import datetime

class SplitData(db.Model):
    __tablename__ = 'split_data'
    
    id_split = db.Column(db.Integer, primary_key=True)
    id_data_citra = db.Column(db.Integer, db.ForeignKey('data_citra.id_data_citra'), nullable=False)
    id_kategori = db.Column(db.Integer, db.ForeignKey('kategori_penyakit.id_kategori'), nullable=False)
    jenis_split = db.Column(db.Enum('train', 'test', name='jenis_split_enum'), nullable=False)
    tanggal_split = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<SplitData {self.id_split} - {self.jenis_split}>'
    
    def to_dict(self):
        return {
            'id_split': self.id_split,
            'id_data_citra': self.id_data_citra,
            'id_kategori': self.id_kategori,
            'jenis_split': self.jenis_split,
            'tanggal_split': self.tanggal_split.isoformat() if self.tanggal_split else None
        }