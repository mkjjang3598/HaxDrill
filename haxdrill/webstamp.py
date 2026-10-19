"""Bump the ?v= cache-buster on every local <script> in web/index.html (run after editing web/*.js).

    python -m haxdrill.webstamp [web/index.html]
"""
import re, sys, time

path = sys.argv[1] if len(sys.argv) > 1 else "web/index.html"
v = time.strftime("%Y%m%d%H%M%S")
s = open(path).read()
s2 = re.sub(r'<script src="([a-z0-9_]+\.js)(\?v=[^"]*)?"', lambda m: f'<script src="{m.group(1)}?v={v}"', s)
open(path, "w").write(s2)
print(f"stamped {path} with v={v}")
