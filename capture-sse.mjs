import * as http from 'node:http';
import * as fs from 'node:fs';
import * as path from 'node:path';

const PORT = parseInt(process.argv[2] || process.env.OPENCODE_SERVER_PORT || '0', 10);
const PASSWORD = process.argv[3] || process.env.OPENCODE_SERVER_PASSWORD || '';
const USERNAME = process.argv[4] || process.env.OPENCODE_SERVER_USERNAME || 'opencode';

if (!PORT || !PASSWORD) {
  console.error(`Usage: node capture-sse.mjs <port> <password> [username]`);
  console.error(`Or set OPENCODE_SERVER_PORT, OPENCODE_SERVER_PASSWORD, OPENCODE_SERVER_USERNAME`);
  process.exit(1);
}

const AUTH = Buffer.from(`${USERNAME}:${PASSWORD}`).toString('base64');
const LOGFILE = path.join(process.env.TEMP || 'C:\\Temp', 'opencode', `sse-events-${PORT}.log`);
const logDir = path.dirname(LOGFILE);
if (!fs.existsSync(logDir)) fs.mkdirSync(logDir, { recursive: true });

function log(obj) {
  const line = JSON.stringify({ ts: new Date().toISOString(), ...obj }) + '\n';
  fs.appendFileSync(LOGFILE, line);
  process.stdout.write(line);
}

log({ msg: 'START', port: PORT, user: USERNAME, log: LOGFILE });

const req = http.request({
  hostname: '127.0.0.1',
  port: PORT,
  path: '/global/event',
  method: 'GET',
  headers: {
    'Authorization': `Basic ${AUTH}`,
    'Accept': 'text/event-stream',
    'Cache-Control': 'no-cache',
  },
  timeout: 0,
});

req.on('response', (res) => {
  log({ msg: 'CONNECTED', status: res.statusCode });

  let buf = '';
  let eventType = '', eventData = '', eventId = '';

  res.setEncoding('utf8');
  res.on('data', (chunk) => {
    buf += chunk;
    const lines = buf.split('\n');
    buf = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.trimEnd();
      if (trimmed.startsWith('id: ')) {
        eventId = trimmed.slice(4);
      } else if (trimmed.startsWith('event: ')) {
        eventType = trimmed.slice(7);
      } else if (trimmed.startsWith('data: ')) {
        eventData += trimmed.slice(6);
      } else if (trimmed === '') {
        if (eventType || eventData) {
          const parsed = { id: eventId, event: eventType };
          if (eventData) {
            try { parsed.data = JSON.parse(eventData); } catch { parsed.data = eventData; }
          }
          log({ msg: 'EVENT', ...parsed });
        }
        eventType = ''; eventData = ''; eventId = '';
      }
    }
  });

  res.on('end', () => {
    log({ msg: 'DISCONNECTED' });
    process.exit(0);
  });

  res.on('error', (err) => {
    log({ msg: 'STREAM_ERROR', error: err.message });
  });
});

req.on('error', (err) => {
  log({ msg: 'FAILED', error: err.message });
  process.exit(1);
});

req.end();

let running = true;
process.on('SIGINT', () => {
  if (!running) return;
  running = false;
  log({ msg: 'STOPPED' });
  process.exit(0);
});
process.on('SIGTERM', () => {
  log({ msg: 'TERMINATED' });
  process.exit(0);
});
