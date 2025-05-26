from app.extension import db
from datetime import datetime

class DataCitra(db.Model):
    __tablename__ = 'data_citra'
    
    id_data_citra = db.Column(db.Integer, primary_key=True)
    id_gambar = db.Column(db.Integer, db.ForeignKey('dataset_gambar.id_gambar'), nullable=False)
    contrast = db.Column(db.Float)
    dissimilarity = db.Column(db.Float)
    homogeneity = db.Column(db.Float)
    energy = db.Column(db.Float)
    correlation = db.Column(db.Float)
    asm = db.Column(db.Float)
    tanggal_upload = db.Column(db.DateTime, default=datetime.utcnow)
    uploaded_by = db.Column(db.Integer, db.ForeignKey('users.id_user'))
    
    def __repr__(self):
        return f'<DataCitra {self.id_data_citra}>'
    
    def to_dict(self):
        return {
            'id_data_citra': self.id_data_citra,
            'id_gambar': self.id_gambar,
            'contrast': self.contrast,
            'dissimilarity': self.dissimilarity,
            'homogeneity': self.homogeneity,
            'energy': self.energy,
            'correlation': self.correlation,
            'asm': self.asm,
            'tanggal_upload': self.tanggal_upload.isoformat() if self.tanggal_upload else None,
            'uploaded_by': self.uploaded_by
        }