from flask import Flask, session
from app.extension import db
from app.models.AuthModel import User
from datetime import datetime, timedelta

def create_app():
    from routes import register_routes

    app = Flask(__name__)
    app.config.from_object('config.Config')

    db.init_app(app)
    register_routes(app)

    @app.context_processor
    def inject_globals():
        user = None
        greeting = ''
        if 'user_id' in session:
            user = User.query.get(session['user_id'])

            now = datetime.utcnow() + timedelta(hours=7)
            hour = now.hour

            if 5 <= hour < 12:
                greeting = "Selamat Pagi"
            elif 12 <= hour < 18:
                greeting = "Selamat Siang"
            elif 18 <= hour < 22:
                greeting = "Selamat Sore"
            else:
                greeting = "Selamat Malam"

        return {'user': user, 'greeting': greeting}

    return app
