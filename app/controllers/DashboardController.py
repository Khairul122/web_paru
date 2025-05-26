from flask import Blueprint, render_template, session, redirect, url_for
from datetime import datetime, timedelta
from app.models.AuthModel import User

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/dashboard')
def index():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))

    user = User.query.get(session['user_id'])

    now = datetime.utcnow() + timedelta(hours=7)
    current_hour = now.hour

    if 5 <= current_hour < 12:
        greeting = "Selamat Pagi"
    elif 12 <= current_hour < 18:
        greeting = "Selamat Siang"
    elif 18 <= current_hour < 22:
        greeting = "Selamat Sore"
    else:
        greeting = "Selamat Malam"

    return render_template(
        'views/dashboard/index.html',
        user=user,
        greeting=greeting,
        user_name=user.nama_lengkap  # kolom di database
    )
