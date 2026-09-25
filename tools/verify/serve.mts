import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const ROOT = process.argv[2] || process.cwd();
const PORT = Number(process.argv[3] || 8931);

const MIME: Record<string, string> = {
  '.js': 'text/javascript',
  '.html': 'text/html',
  '.json': 'application/json',
};

const server = http.createServer((req, res) => {
  try {
    const url = new URL(req.url || '/', 'http://localhost');
    let p = path.join(ROOT, decodeURIComponent(url.pathname));
    if (p.endsWith('/')) p = path.join(p, 'index.html');
    if (!fs.existsSync(p) || !fs.statSync(p).isFile()) {
      res.writeHead(404);
      res.end('not found: ' + p);
      return;
    }
    const ext = path.extname(p);
    res.writeHead(200, { 'Content-Type': MIME[ext] || 'application/octet-stream' });
    fs.createReadStream(p).pipe(res);
  } catch (e) {
    res.writeHead(500);
    res.end(String(e));
  }
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`serving ${ROOT} on http://127.0.0.1:${PORT}`);
});
