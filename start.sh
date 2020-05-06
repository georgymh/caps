exec gunicorn --bind 0.0.0.0:8080 wsgi:app --max-requests 10000 --timeout 10 --keep-alive 10 --log-level info
