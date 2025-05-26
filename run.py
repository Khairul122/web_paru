from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)


### Struktur file: app/__init__.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from app.controllers.auth import auth_bp

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.secret_key = 'secretkeyanda'

    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)

    app.register_blueprint(auth_bp, url_prefix='')

    return app
