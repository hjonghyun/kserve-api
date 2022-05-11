 #!/bin/sh
 gunicorn --chdir app --worker-class gevent --workers 4 --bind 0.0.0.0:5000 kserve_api:app --max-requests 10000 --timeout 30 --keep-alive 10 --log-level debug