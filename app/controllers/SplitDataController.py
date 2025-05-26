from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify, make_response
from app.extension import db
from app.models.DataCitraModel import DataCitra
from app.models.DatasetModel import DatasetGambar
from app.models.KategoriModel import KategoriPenyakit
from app.models.SplitDataModel import SplitData
from sqlalchemy import func, desc, asc
import random
import csv
import io
from datetime import datetime

split_bp = Blueprint('split', __name__)

def get_pagination_info(page, per_page, total):
    total_pages = (total + per_page - 1) // per_page
    has_prev = page > 1
    has_next = page < total_pages
    prev_num = page - 1 if has_prev else None
    next_num = page + 1 if has_next else None
    
    return {
        'page': page,
        'per_page': per_page,
        'total': total,
        'total_pages': total_pages,
        'has_prev': has_prev,
        'has_next': has_next,
        'prev_num': prev_num,
        'next_num': next_num
    }

def get_split_statistics():
    try:
        total_split = SplitData.query.count()
        total_train = SplitData.query.filter_by(jenis_split='train').count()
        total_test = SplitData.query.filter_by(jenis_split='test').count()
        total_glcm = DataCitra.query.count()
        total_unsplit = total_glcm - total_split
        
        print(f"DEBUG STATS: split={total_split}, train={total_train}, test={total_test}, glcm={total_glcm}, unsplit={total_unsplit}")
        
        return {
            'total_split': total_split,
            'total_train': total_train,
            'total_test': total_test,
            'total_unsplit': total_unsplit,
            'total_glcm': total_glcm,
            'kategori_stats': []
        }
    except Exception as e:
        print(f"ERROR in get_split_statistics: {e}")
        return {
            'total_split': 0,
            'total_train': 0,
            'total_test': 0,
            'total_unsplit': 0,
            'total_glcm': 0,
            'kategori_stats': []
        }
    
@split_bp.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    kategori_filter = request.args.get('kategori', '', type=str)
    jenis_filter = request.args.get('jenis', '', type=str)
    sort_by = request.args.get('sort', 'tanggal_split', type=str)
    sort_order = request.args.get('order', 'desc', type=str)
    
    if per_page not in [5, 10, 25, 50, 100]:
        per_page = 10
    
    query = db.session.query(
        SplitData,
        DataCitra,
        DatasetGambar,
        KategoriPenyakit
    ).join(
        DataCitra, SplitData.id_data_citra == DataCitra.id_data_citra
    ).join(
        DatasetGambar, DataCitra.id_gambar == DatasetGambar.id_gambar
    ).join(
        KategoriPenyakit, SplitData.id_kategori == KategoriPenyakit.id_kategori
    )
    
    if kategori_filter:
        query = query.filter(KategoriPenyakit.id_kategori == kategori_filter)
    
    if jenis_filter:
        query = query.filter(SplitData.jenis_split == jenis_filter)
    
    if sort_by == 'tanggal_split':
        if sort_order == 'desc':
            query = query.order_by(desc(SplitData.tanggal_split))
        else:
            query = query.order_by(asc(SplitData.tanggal_split))
    elif sort_by == 'kategori':
        if sort_order == 'desc':
            query = query.order_by(desc(KategoriPenyakit.nama_kategori))
        else:
            query = query.order_by(asc(KategoriPenyakit.nama_kategori))
    elif sort_by == 'jenis_split':
        if sort_order == 'desc':
            query = query.order_by(desc(SplitData.jenis_split))
        else:
            query = query.order_by(asc(SplitData.jenis_split))
    elif sort_by == 'nama_file':
        if sort_order == 'desc':
            query = query.order_by(desc(DatasetGambar.nama_file))
        else:
            query = query.order_by(asc(DatasetGambar.nama_file))
    
    total = query.count()
    offset = (page - 1) * per_page
    split_data_list = query.offset(offset).limit(per_page).all()
    
    pagination = get_pagination_info(page, per_page, total)
    stats = get_split_statistics()
    kategori_list = KategoriPenyakit.query.all()
    
    return render_template('views/split/index.html', 
                         split_data_list=split_data_list,
                         pagination=pagination,
                         stats=stats,
                         kategori_list=kategori_list,
                         current_filters={
                             'kategori': kategori_filter,
                             'jenis': jenis_filter,
                             'sort': sort_by,
                             'order': sort_order,
                             'per_page': per_page
                         })

@split_bp.route('/create')
def create():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    available_data = db.session.query(
        DataCitra,
        DatasetGambar,
        KategoriPenyakit
    ).join(
        DatasetGambar, DataCitra.id_gambar == DatasetGambar.id_gambar
    ).join(
        KategoriPenyakit, DatasetGambar.id_kategori == KategoriPenyakit.id_kategori
    ).filter(
        ~DataCitra.id_data_citra.in_(
            db.session.query(SplitData.id_data_citra)
        )
    ).all()
    
    kategori_stats = db.session.query(
        KategoriPenyakit.id_kategori,
        KategoriPenyakit.nama_kategori,
        func.count(DataCitra.id_data_citra).label('total')
    ).join(
        DatasetGambar, KategoriPenyakit.id_kategori == DatasetGambar.id_kategori
    ).join(
        DataCitra, DatasetGambar.id_gambar == DataCitra.id_gambar
    ).filter(
        ~DataCitra.id_data_citra.in_(
            db.session.query(SplitData.id_data_citra)
        )
    ).group_by(
        KategoriPenyakit.id_kategori,
        KategoriPenyakit.nama_kategori
    ).all()
    
    return render_template('views/split/create.html',
                         available_data=available_data,
                         kategori_stats=kategori_stats)

@split_bp.route('/process', methods=['POST'])
def process():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    try:
        split_method = request.form.get('split_method', 'percentage')
        train_percentage = float(request.form.get('train_percentage', 80))
        
        if split_method == 'percentage':
            result = process_percentage_split(train_percentage)
        elif split_method == 'manual':
            selected_train = request.form.getlist('train_data')
            selected_test = request.form.getlist('test_data')
            result = process_manual_split(selected_train, selected_test)
        elif split_method == 'stratified':
            result = process_stratified_split(train_percentage)
        else:
            flash('Metode split tidak valid', 'error')
            return redirect(url_for('split.create'))
        
        if result['success']:
            flash(f"Split data berhasil! Train: {result['train_count']}, Test: {result['test_count']}", 'success')
        else:
            flash(f"Split data gagal: {result['message']}", 'error')
        
        return redirect(url_for('split.index'))
    
    except Exception as e:
        db.session.rollback()
        flash(f'Terjadi error: {str(e)}', 'error')
        return redirect(url_for('split.create'))

def process_percentage_split(train_percentage):
    try:
        available_data = db.session.query(DataCitra, DatasetGambar).join(
            DatasetGambar, DataCitra.id_gambar == DatasetGambar.id_gambar
        ).filter(
            ~DataCitra.id_data_citra.in_(
                db.session.query(SplitData.id_data_citra)
            )
        ).all()
        
        if not available_data:
            return {'success': False, 'message': 'Tidak ada data yang tersedia untuk di-split'}
        
        data_list = list(available_data)
        random.shuffle(data_list)
        
        total_data = len(data_list)
        train_count = int(total_data * train_percentage / 100)
        
        train_data = data_list[:train_count]
        test_data = data_list[train_count:]
        
        for data_citra, dataset in train_data:
            split_data = SplitData(
                id_data_citra=data_citra.id_data_citra,
                id_kategori=dataset.id_kategori,
                jenis_split='train'
            )
            db.session.add(split_data)
        
        for data_citra, dataset in test_data:
            split_data = SplitData(
                id_data_citra=data_citra.id_data_citra,
                id_kategori=dataset.id_kategori,
                jenis_split='test'
            )
            db.session.add(split_data)
        
        db.session.commit()
        
        return {
            'success': True,
            'train_count': len(train_data),
            'test_count': len(test_data)
        }
    
    except Exception as e:
        db.session.rollback()
        return {'success': False, 'message': str(e)}

def process_stratified_split(train_percentage):
    try:
        kategori_data = db.session.query(
            KategoriPenyakit.id_kategori,
            KategoriPenyakit.nama_kategori
        ).all()
        
        total_train = 0
        total_test = 0
        
        for kategori in kategori_data:
            kategori_available = db.session.query(DataCitra, DatasetGambar).join(
                DatasetGambar, DataCitra.id_gambar == DatasetGambar.id_gambar
            ).filter(
                DatasetGambar.id_kategori == kategori.id_kategori,
                ~DataCitra.id_data_citra.in_(
                    db.session.query(SplitData.id_data_citra)
                )
            ).all()
            
            if not kategori_available:
                continue
            
            kategori_list = list(kategori_available)
            random.shuffle(kategori_list)
            
            kategori_total = len(kategori_list)
            kategori_train_count = int(kategori_total * train_percentage / 100)
            
            train_data = kategori_list[:kategori_train_count]
            test_data = kategori_list[kategori_train_count:]
            
            for data_citra, dataset in train_data:
                split_data = SplitData(
                    id_data_citra=data_citra.id_data_citra,
                    id_kategori=kategori.id_kategori,
                    jenis_split='train'
                )
                db.session.add(split_data)
                total_train += 1
            
            for data_citra, dataset in test_data:
                split_data = SplitData(
                    id_data_citra=data_citra.id_data_citra,
                    id_kategori=kategori.id_kategori,
                    jenis_split='test'
                )
                db.session.add(split_data)
                total_test += 1
        
        db.session.commit()
        
        return {
            'success': True,
            'train_count': total_train,
            'test_count': total_test
        }
    
    except Exception as e:
        db.session.rollback()
        return {'success': False, 'message': str(e)}

def process_manual_split(selected_train, selected_test):
    try:
        if not selected_train and not selected_test:
            return {'success': False, 'message': 'Pilih minimal satu data untuk train atau test'}
        
        train_set = set(selected_train)
        test_set = set(selected_test)
        
        if train_set & test_set:
            return {'success': False, 'message': 'Ada data yang dipilih untuk train dan test sekaligus'}
        
        for id_data_citra in selected_train:
            data_citra = DataCitra.query.get(id_data_citra)
            if data_citra:
                dataset = DatasetGambar.query.get(data_citra.id_gambar)
                split_data = SplitData(
                    id_data_citra=int(id_data_citra),
                    id_kategori=dataset.id_kategori,
                    jenis_split='train'
                )
                db.session.add(split_data)
        
        for id_data_citra in selected_test:
            data_citra = DataCitra.query.get(id_data_citra)
            if data_citra:
                dataset = DatasetGambar.query.get(data_citra.id_gambar)
                split_data = SplitData(
                    id_data_citra=int(id_data_citra),
                    id_kategori=dataset.id_kategori,
                    jenis_split='test'
                )
                db.session.add(split_data)
        
        db.session.commit()
        
        return {
            'success': True,
            'train_count': len(selected_train),
            'test_count': len(selected_test)
        }
    
    except Exception as e:
        db.session.rollback()
        return {'success': False, 'message': str(e)}

@split_bp.route('/detail/<int:id>')
def detail(id):
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    split_data = db.session.query(
        SplitData,
        DataCitra,
        DatasetGambar,
        KategoriPenyakit
    ).join(
        DataCitra, SplitData.id_data_citra == DataCitra.id_data_citra
    ).join(
        DatasetGambar, DataCitra.id_gambar == DatasetGambar.id_gambar
    ).join(
        KategoriPenyakit, SplitData.id_kategori == KategoriPenyakit.id_kategori
    ).filter(SplitData.id_split == id).first()
    
    if not split_data:
        flash('Data tidak ditemukan', 'error')
        return redirect(url_for('split.index'))
    
    return render_template('views/split/detail.html', split_data=split_data)

@split_bp.route('/hapus/<int:id>')
def hapus(id):
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    try:
        split_data = SplitData.query.get_or_404(id)
        db.session.delete(split_data)
        db.session.commit()
        
        flash('Data split berhasil dihapus', 'success')
        return redirect(url_for('split.index'))
    
    except Exception as e:
        db.session.rollback()
        flash(f'Gagal menghapus data: {str(e)}', 'error')
        return redirect(url_for('split.index'))

@split_bp.route('/reset', methods=['POST'])
def reset():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    try:
        deleted_count = SplitData.query.count()
        SplitData.query.delete()
        db.session.commit()
        
        flash(f'Berhasil reset {deleted_count} data split', 'success')
        return redirect(url_for('split.index'))
    
    except Exception as e:
        db.session.rollback()
        flash(f'Gagal reset data split: {str(e)}', 'error')
        return redirect(url_for('split.index'))

@split_bp.route('/export_csv')
def export_csv():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    try:
        jenis_export = request.args.get('jenis', 'all')
        
        query = db.session.query(
            SplitData,
            DataCitra,
            DatasetGambar,
            KategoriPenyakit
        ).join(
            DataCitra, SplitData.id_data_citra == DataCitra.id_data_citra
        ).join(
            DatasetGambar, DataCitra.id_gambar == DatasetGambar.id_gambar
        ).join(
            KategoriPenyakit, SplitData.id_kategori == KategoriPenyakit.id_kategori
        )
        
        if jenis_export in ['train', 'test']:
            query = query.filter(SplitData.jenis_split == jenis_export)
        
        split_data_list = query.all()
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        writer.writerow([
            'ID Split', 'Jenis Split', 'Nama File', 'Kategori',
            'Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'ASM',
            'Tanggal Split'
        ])
        
        for split_data, data_citra, dataset, kategori in split_data_list:
            writer.writerow([
                split_data.id_split,
                split_data.jenis_split,
                dataset.nama_file,
                kategori.nama_kategori,
                data_citra.contrast,
                data_citra.dissimilarity,
                data_citra.homogeneity,
                data_citra.energy,
                data_citra.correlation,
                data_citra.asm,
                split_data.tanggal_split.strftime('%Y-%m-%d %H:%M:%S')
            ])
        
        output.seek(0)
        
        filename = f'split_data_{jenis_export}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename={filename}'
        
        return response
    
    except Exception as e:
        flash(f'Gagal export CSV: {str(e)}', 'error')
        return redirect(url_for('split.index'))

@split_bp.route('/api/stats')
def api_stats():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        stats = get_split_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@split_bp.route('/test')
def test():
    try:
        total_glcm = DataCitra.query.count()
        total_split = SplitData.query.count()
        
        available_count = db.session.query(DataCitra).filter(
            ~DataCitra.id_data_citra.in_(
                db.session.query(SplitData.id_data_citra)
            )
        ).count()
        
        return {
            'total_glcm': total_glcm,
            'total_split': total_split,
            'available_for_split': available_count,
            'message': 'Test successful'
        }
    except Exception as e:
        import traceback
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }