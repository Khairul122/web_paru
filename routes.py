from flask import Flask, redirect, url_for
from app.controllers.AuthController import auth_bp
from app.controllers.DashboardController import dashboard_bp
from app.controllers.KategoriController import kategori_bp
from app.controllers.DatasetController import dataset_bp
from app.controllers.GLCMController import glcm_bp
from app.controllers.SplitDataController import split_bp
from app.controllers.LDAController import lda_bp
from app.controllers.PrediksiController import prediksi_bp

def register_routes(app: Flask):
    @app.route('/')
    def root_redirect():
        return redirect(url_for('auth.login'))  

    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(dashboard_bp, url_prefix='/dashboard')
    app.register_blueprint(kategori_bp, url_prefix='/kategori')
    app.register_blueprint(dataset_bp, url_prefix='/dataset')
    app.register_blueprint(glcm_bp, url_prefix='/glcm')
    app.register_blueprint(split_bp, url_prefix='/split')
    app.register_blueprint(lda_bp, url_prefix='/lda')
    app.register_blueprint(prediksi_bp, url_prefix='/prediksi')