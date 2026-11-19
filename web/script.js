const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

document.querySelectorAll('.tab').forEach(b => b.onclick = () => {
  document.querySelectorAll('.tab,.view').forEach(x => x.classList.remove('active'));
  b.classList.add('active');
  $('#' + b.dataset.view).classList.add('active');
  if (b.dataset.view === 'convert') { loadConversions(); loadHealth(); }
  if (b.dataset.view === 'storage') { loadStorage(); }
});

async function api(url, opt) {
  const r = await fetch(url, opt);
  const d = await r.json();
  if (!r.ok) throw Error(d.detail || '请求失败');
  return d;
}

let modelConfig = null;

async function loadConfig() {
  modelConfig = await api('/api/config');
  const chats = modelConfig.models.filter(x => ['chat'].includes(x.api_style.toLowerCase()));
  const embeds = modelConfig.models.filter(x => ['embedding'].includes(x.api_style.toLowerCase()));
  $('#chat-model').innerHTML = chats.map(x => `<option value="${esc(x.name)}">${esc(x.name)}</option>`).join('');
  $('#embedding-model').innerHTML = embeds.map(x => `<option value="${esc(x.name)}">${esc(x.name)}</option>`).join('');
  $('#chat-model').value = modelConfig.chat_model;
  $('#embedding-model').value = modelConfig.embedding_model;
}

async function saveModels() {
  try {
    const d = await api('/api/settings/models', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        chat_model: $('#chat-model').value,
        embedding_model: $('#embedding-model').value
      })
    });
    $('#model-status').textContent = `已保存：${d.chat_model} / ${d.embedding_model}`;
  } catch (e) {
    $('#model-status').textContent = '保存失败：' + e.message;
  }
}

async function refreshAll() {
  api('/api/storage').then(d => {
    if ($('#app-version')) $('#app-version').textContent = d.version || '';
  }).catch(() => {});
  await Promise.allSettled([loadConfig(), loadHealth(), loadUsage(), loadDocuments(), loadIngestTasks()]);
}

async function loadHealth() {
  const el = $('#health');
  el.textContent = '检查中…';
  try {
    const d = await api('/api/health');
    el.innerHTML = [...d.models, ...d.services].map(x => {
      const note = [x.category || '服务',
        x.latency_ms ? `${x.latency_ms} ms` : '',
        x.detail || '',
        x.error || ''].filter(Boolean).join(' · ');
      return `<div class="row">
        <div>
          <b>${esc(x.name)}</b>
          <div class="muted">${esc(note)}</div>
        </div>
        <span class="status ${x.status === 'connected' ? 'connected' : 'error'}">${x.status === 'connected' ? '已连接' : '异常'}</span>
      </div>`;
    }).join('');
    const mineru = d.services.find(x => x.name === 'MinerU');
    if (mineru) updateMineruState(mineru);
  } catch (e) {
    el.textContent = e.message;
  }
}

function updateMineruState(mineru) {
  const el = $('#mineru-state');
  if (!el) return;
  if (mineru.status === 'connected') {
    const bits = [`MinerU ${mineru.version || ''} 已连接`];
    if (mineru.protocol_version) bits.push(`协议 v${mineru.protocol_version}`);
    if (mineru.concurrency) bits.push(`并发上限 ${mineru.concurrency}`);
    if (mineru.queued_tasks !== undefined && mineru.queued_tasks !== null) {
      bits.push(`队列 ${mineru.queued_tasks} · 处理中 ${mineru.processing_tasks || 0}`);
    }
    el.textContent = bits.join(' · ');
  } else if (mineru.status === 'not_configured') {
    el.textContent = 'MinerU 未配置（MINERU_URL 为空），PDF/图片将回退到 MarkItDown；Word/Markdown 等仍可转换。';
  } else {
    el.textContent = `MinerU 连接异常：${mineru.error || '未知错误'}；PDF/图片将回退到 MarkItDown。`;
  }
}

async function loadUsage() {
  try {
    const d = await api('/api/usage');
    $('#stat-calls').textContent = d.calls;
    $('#stat-tokens').textContent = Number(d.total_tokens).toLocaleString();
    $('#usage-models').innerHTML = d.models.map(x =>
      `<div class="row">
        <b>${esc(x.model)}</b>
        <span>${x.calls} 次 · ${Number(x.total_tokens).toLocaleString()} tokens · 成功 ${x.successes}</span>
      </div>`
    ).join('') || '暂无调用';
    $('#usage-recent').innerHTML =
      `<table>
        <thead>
          <tr>
            <th>时间</th>
            <th>模型</th>
            <th>任务</th>
            <th>输入/输出</th>
            <th>耗时</th>
            <th>状态</th>
          </tr>
        </thead>
        <tbody>
          ${d.recent.map(x =>
            `<tr>
              <td>${esc(x.created_at)}</td>
              <td>${esc(x.model)}</td>
              <td>${esc(x.task)}</td>
              <td>${x.input_tokens}/${x.output_tokens}${x.estimated ? ' (估算/缺失)' : ''}</td>
              <td>${x.latency_ms} ms</td>
              <td>${x.success ? '成功' : esc(x.error)}</td>
            </tr>`
          ).join('')}
        </tbody>
      </table>`;
  } catch (e) {
    $('#usage-models').textContent = e.message;
  }
}

async function loadDocuments() {
  try {
    const ds = await api('/api/documents');
    $('#stat-docs').textContent = ds.filter(x => x.status === 'ready').length;
    $('#documents-list').innerHTML = ds.map(x =>
      `<div class="row">
        <div>
          <b>${esc(x.name)}</b>
          <div class="muted">${x.chunks} 个片段 · ${esc(x.created_at)} · 切分 ${x.chunk_size} / 重叠 ${x.chunk_overlap} · BGE ${esc(x.embedding_model || '旧索引')} · ${esc(x.collection_name || '默认集合')}${x.error ? ' · ' + esc(x.error) : ''}</div>
        </div>
        <div class="doc-actions">
          <span class="status ${x.status === 'ready' ? 'ready' : 'error'}">${esc(x.status)}</span>
          <button onclick="reindexDoc('${esc(x.id)}',${x.chunk_size},${x.chunk_overlap})">重建</button>
          <button class="danger" onclick="deleteDoc('${esc(x.id)}')">删除</button>
        </div>
      </div>`
    ).join('') || '<div class="muted">还没有文档</div>';
  } catch (e) {
    $('#documents-list').textContent = e.message;
  }
}

async function loadIngestTasks() {
  try {
    const tasks = await api('/api/ingest-tasks');
    $('#ingest-tasks').innerHTML = tasks.map(t =>
      `<div class="card" style="margin:10px 0;padding:13px">
        <b>${esc(t.filename)}</b>
        <span class="status ${esc(t.status)}">${esc(t.status)}</span>
        <div class="muted">任务 ${esc(t.id)} · ${esc(t.updated_at)}</div>
        ${t.details.map(s =>
          `<div class="row">
            <span>${esc(s.label)}</span>
            <span class="status ${esc(s.status)}">${esc(s.status)}</span>
            <small>${esc(s.message)}</small>
          </div>`
        ).join('')}
        ${t.error ? `<div class="error">${esc(t.error)}</div>` : ''}
      </div>`
    ).join('') || '<div class="muted">暂无处理任务</div>';
  } catch (e) {
    $('#ingest-tasks').textContent = e.message;
  }
}

async function uploadFiles() {
  const files = $('#files').files;
  if (!files.length) return alert('请选择文件');
  const size = Number($('#chunk-size').value), overlap = Number($('#chunk-overlap').value);
  if (size < 100 || overlap < 0 || overlap >= size) return alert('请检查切分参数：长度至少 100，重叠须小于长度');
  const status = $('#upload-status');
  for (const file of files) {
    status.textContent = `正在接收 ${file.name}…`;
    const form = new FormData();
    form.append('file', file);
    form.append('chunk_size', size);
    form.append('chunk_overlap', overlap);
    try {
      const d = await api('/api/documents', { method: 'POST', body: form });
      status.textContent = `任务 ${d.task_id} 已提交；Embedding=${d.embedding_model}，collection=${d.collection_name}`;
    } catch (e) {
      status.textContent += ` 失败：${e.message}`;
    }
  }
  await loadDocuments();
  await loadIngestTasks();
  pollTasks();
}

let taskPoll = null;

async function pollTasks() {
  if (taskPoll) clearInterval(taskPoll);
  taskPoll = setInterval(async () => {
    await loadIngestTasks();
    const el = $('#ingest-tasks');
    try {
      const tasks = await api('/api/ingest-tasks');
      if (!tasks.some(t => t.status === 'processing')) {
        clearInterval(taskPoll);
        taskPoll = null;
        await loadDocuments();
        await loadUsage();
      }
    } catch (e) {
      console.warn(e);
    }
  }, 1200);
}

async function reindexDoc(id, size, overlap) {
  const chunkSize = Number(prompt('新的切分长度（字符）', size));
  if (!chunkSize) return;
  const chunkOverlap = Number(prompt('新的切分重叠长度（字符）', overlap));
  if (!Number.isFinite(chunkOverlap) || chunkOverlap >= chunkSize || chunkOverlap < 0) return alert('重叠长度必须小于切分长度且不能为负');
  try {
    await api(`/api/documents/${id}/reindex`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chunk_size: chunkSize, chunk_overlap: chunkOverlap })
    });
    await loadDocuments();
  } catch (e) {
    alert('重建失败：' + e.message);
  }
}

async function deleteDoc(id) {
  if (!confirm('删除文档及其向量索引？')) return;
  try {
    await api(`/api/documents/${id}`, { method: 'DELETE' });
    await loadDocuments();
  } catch (e) {
    alert('删除失败：' + e.message);
  }
}

async function sendQuestion() {
  const q = $('#question').value.trim();
  if (!q) return;
  $('#chat-answer').textContent = '正在生成…';
  $('#chat-sources').innerHTML = '';
  const useRag = $('#use-rag').checked;
  const body = {
    message: q,
    use_rag: useRag,
    chat_model: $('#chat-model').value,
    embedding_model: $('#embedding-model').value
  };
  if (useRag) body.top_k = Number($('#top-k').value);
  if ($('#temperature').value !== '') body.temperature = Number($('#temperature').value);
  if ($('#max-tokens').value) body.max_tokens = Number($('#max-tokens').value);
  try {
    const d = await api('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    $('#chat-answer').textContent = d.answer || '';
    $('#chat-sources').innerHTML = (d.sources || []).map((x, i) =>
      `<div class="source">
        <b>[来源 ${i + 1}] ${esc(x.doc_name)}</b> · ${Number(x.score || 0).toFixed(3)}<br>
        ${esc(x.content)}
      </div>`
    ).join('');
    await loadUsage();
  } catch (e) {
    $('#chat-answer').textContent = '请求失败：' + e.message;
  }
}

$('#use-rag').addEventListener('change', () => {
  $('#topk-wrap').style.display = $('#use-rag').checked ? 'block' : 'none';
});

/* ------------------------------------------------ 文档转换 (MinerU) */
let convertPoll = null;
const CONVERT_STATUS_LABEL = {
  processing: '处理中',
  converted: '已转换',
  indexing: '入库中',
  indexed: '已入库',
  index_failed: '入库失败',
  failed: '转换失败'
};

// Where each stage sits in the pipeline, so the progress bar is meaningful.
const CONVERT_STAGE_ORDER = ['received', 'saving_original', 'parsing', 'writing', 'converted',
  'indexing', 'indexed'];
const CONVERT_STAGE_LABEL = {
  received: '排队', saving_original: '保存原件', parsing: '解析', writing: '生成文件',
  converted: '完成', indexing: '入库', indexed: '已入库',
  index_failed: '入库失败', failed: '失败'
};

async function loadConversions() {
  try {
    const d = await api('/api/conversions');
    const state = $('#mineru-state');
    if (d.mineru_configured) {
      state.textContent = `MinerU 已启用${d.mineru_async ? '（异步提交 + 轮询）' : '（同步）'}，`
        + `路由格式：${d.mineru_extensions.join(' ') || '仅 PDF/图片'}`;
    } else if (state.textContent.startsWith('MinerU 未配置')) {
      // keep the more detailed not-configured text written by loadHealth
    }
    const rows = d.items;
    $('#conversions-list').innerHTML = rows.map(x => {
      const label = CONVERT_STATUS_LABEL[x.status] || x.status;
      const cls = x.status === 'failed' || x.status === 'index_failed' ? 'error'
        : x.status === 'converted' || x.status === 'indexed' ? 'completed' : 'processing';
      const artifacts = x.json_key
        ? `${x.page_count} 页 · ${x.block_count} 块 · ${Number(x.char_count).toLocaleString()} 字符 · ${x.image_count} 图 · ${esc(x.converter || '')}`
        : esc(x.message || '');
      const canIndex = ['converted', 'index_failed', 'indexed'].includes(x.status);
      const canRetry = ['failed', 'index_failed'].includes(x.status);
      const stageIdx = CONVERT_STAGE_ORDER.indexOf(x.stage);
      const pct = stageIdx < 0 ? 0 : Math.min(100,
        Math.round(((stageIdx + (x.status === 'indexed' || x.status === 'converted' ? 1 : 0)) /
          (CONVERT_STAGE_ORDER.length - 1)) * 100));
      return `<div class="row">
        <div>
          <input type="checkbox" class="convert-pick" value="${esc(x.id)}" ${x.json_key ? '' : 'disabled'}>
          <b>${esc(x.filename)}</b>
          <span class="status ${cls}">${esc(label)}</span>
          ${['processing', 'indexing'].includes(x.status)
            ? `<div class="bar"><span style="width:${pct}%"></span></div>
               <small class="muted">${esc(CONVERT_STAGE_LABEL[x.stage] || x.stage || '')} · ${esc(x.message || '')}</small>`
            : ''}
          <div class="muted">${artifacts} · ${esc(x.updated_at)}</div>
          ${x.error ? `<div class="error">${esc(x.error)}</div>` : ''}
        </div>
        <div class="doc-actions">
          <button onclick="previewConversion('${esc(x.id)}')" ${x.json_key ? '' : 'disabled'}>预览</button>
          <button onclick="indexConversion('${esc(x.id)}')" ${canIndex ? '' : 'disabled'}>继续切分入库</button>
          <button onclick="downloadConversion('${esc(x.id)}')" ${x.json_key ? '' : 'disabled'}>下载</button>
          <button onclick="retryConversion('${esc(x.id)}')" ${canRetry ? '' : 'disabled'}>重试</button>
          <button class="danger" onclick="deleteConversion('${esc(x.id)}')">删除</button>
        </div>
      </div>`;
    }).join('') || '<div class="muted">还没有转换任务</div>';
    if (rows.some(x => ['processing', 'indexing'].includes(x.status))) startConvertPoll();
  } catch (e) {
    $('#conversions-list').textContent = e.message;
  }
}

function startConvertPoll() {
  if (convertPoll) return;
  convertPoll = setInterval(async () => {
    await loadConversions();
    if (!$('#conversions-list').querySelector('.status.processing')) {
      clearInterval(convertPoll);
      convertPoll = null;
      await loadDocuments();
    }
  }, 1500);
}

async function startConversions() {
  const files = $('#convert-files').files;
  if (!files.length) return alert('请选择文件');
  const status = $('#convert-status');
  const form = new FormData();
  for (const f of files) form.append('files', f);
  status.textContent = `正在上传 ${files.length} 个文件…`;
  try {
    const d = await api('/api/conversions', { method: 'POST', body: form });
    const rejected = d.rejected.map(x => `${x.filename}（${x.error}）`).join('；');
    status.textContent = `已提交 ${d.items.length} 个文件，批次 ${d.batch_id}` + (rejected ? `；跳过：${rejected}` : '');
    await loadConversions();
    startConvertPoll();
  } catch (e) {
    status.textContent = '提交失败：' + e.message;
  }
}

async function indexConversion(id) {
  const size = Number($('#convert-chunk-size').value);
  const overlap = Number($('#convert-chunk-overlap').value);
  if (size < 100 || overlap < 0 || overlap >= size) return alert('请检查切分参数：长度至少 100，重叠须小于长度');
  try {
    const d = await api(`/api/conversions/${id}/index`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chunk_size: size, chunk_overlap: overlap })
    });
    $('#convert-status').textContent = `已提交入库任务 ${d.task_id}（Embedding=${d.embedding_model}）`;
    await loadConversions();
    await loadIngestTasks();
    pollTasks();
  } catch (e) {
    alert('入库失败：' + e.message);
  }
}

function downloadConversion(id) {
  window.location.href = `/api/conversions/${id}/download`;
}

async function downloadSelectedConversions() {
  const ids = [...document.querySelectorAll('.convert-pick:checked')].map(x => x.value);
  if (!ids.length) return alert('请先勾选要下载的转换结果');
  try {
    const r = await fetch('/api/conversions/download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ids })
    });
    if (!r.ok) throw Error((await r.json()).detail || '下载失败');
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'conversions.zip';
    a.click();
    URL.revokeObjectURL(url);
  } catch (e) {
    alert('下载失败：' + e.message);
  }
}

async function deleteConversion(id) {
  if (!confirm('删除该转换记录及其 JSON/Markdown 文件？（已入库的向量不受影响）')) return;
  try {
    await api(`/api/conversions/${id}`, { method: 'DELETE' });
    await loadConversions();
  } catch (e) {
    alert('删除失败：' + e.message);
  }
}

async function retryConversion(id) {
  try {
    await api(`/api/conversions/${id}/retry`, { method: 'POST' });
    $('#convert-status').textContent = '已重新提交解析';
    await loadConversions();
    startConvertPoll();
  } catch (e) {
    alert('重试失败：' + e.message);
  }
}

async function previewConversion(id) {
  const panel = $('#conversion-preview');
  panel.hidden = false;
  panel.innerHTML = '<div class="muted">正在读取转换结果…</div>';
  try {
    const d = await api(`/api/conversions/${id}/preview`);
    const outline = d.outline.length
      ? `<div class="outline">${d.outline.map(h =>
          `<div style="padding-left:${Math.min(h.level - 1, 5) * 14}px">${esc(h.text)}</div>`
        ).join('')}</div>`
      : '<div class="muted">没有检测到标题层级</div>';
    panel.innerHTML = `
      <div class="row">
        <b>转换结果预览</b>
        <button onclick="closePreview()">关闭</button>
      </div>
      <div class="muted">${d.page_count} 页 · 共 ${Number(d.total_chars).toLocaleString()} 字符${d.truncated ? ' · 仅显示开头' : ''}</div>
      <details open><summary>标题结构</summary>${outline}</details>
      <details><summary>Markdown 内容</summary><pre class="preview-md">${esc(d.markdown)}</pre></details>`;
  } catch (e) {
    panel.innerHTML = `<div class="error">预览失败：${esc(e.message)}</div>
      <button onclick="closePreview()">关闭</button>`;
  }
}

function closePreview() {
  const panel = $('#conversion-preview');
  panel.hidden = true;
  panel.innerHTML = '';
}

/* ------------------------------------------------ 存储设置 (MinIO / Milvus) */
function fillStorage(d) {
  $('#milvus-host').value = d.milvus.host;
  $('#milvus-port').value = d.milvus.port;
  $('#minio-host').value = d.minio.host;
  $('#minio-port').value = d.minio.port;
  $('#milvus-summary').textContent =
    `${d.milvus.uri} · 数据库 ${d.milvus.database || 'default'} · collection ${d.milvus.collection}`;
  $('#minio-summary').textContent =
    `${d.minio.endpoint} · bucket ${d.minio.bucket}`;
  $('#milvus-link').href = d.milvus.console_url;
  $('#milvus-link').textContent = `打开 Milvus 管理界面（${d.milvus.console_url}）→`;
  $('#minio-link').href = d.minio.console_url;
  $('#minio-link').textContent = `打开 MinIO 管理界面（${d.minio.console_url}）→`;
}

async function loadStorage() {
  try {
    const d = await api('/api/storage');
    fillStorage(d);
    if ($('#app-version')) $('#app-version').textContent = d.version || '';
  } catch (e) {
    $('#storage-status').textContent = '读取失败：' + e.message;
  }
}

function storageBody() {
  return {
    milvus_host: $('#milvus-host').value.trim(),
    milvus_port: Number($('#milvus-port').value),
    minio_host: $('#minio-host').value.trim(),
    minio_port: Number($('#minio-port').value)
  };
}

async function saveStorage() {
  const body = storageBody();
  if (!body.milvus_host || !body.minio_host) return alert('请填写 IP / 主机名');
  for (const [k, v] of Object.entries(body)) {
    if (k.endsWith('_port') && (!Number.isInteger(v) || v < 1 || v > 65535)) {
      return alert('端口必须是 1-65535 的整数');
    }
  }
  try {
    fillStorage(await api('/api/storage', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    }));
    $('#storage-status').textContent = '已保存并生效';
  } catch (e) {
    $('#storage-status').textContent = '保存失败：' + e.message;
  }
}

function probeRow(name, d) {
  const ok = d.status === 'connected';
  const detail = ok
    ? [d.uri || d.endpoint, d.database ? `数据库 ${d.database}` : '', d.bucket ? `bucket ${d.bucket}` : '',
       d.collections ? `${d.collections.length} 个 collection` : ''].filter(Boolean).join(' · ')
    : (d.error || '连接失败');
  return `<div class="row">
    <div><b>${esc(name)}</b><div class="muted">${esc(detail)}</div></div>
    <span class="status ${ok ? 'connected' : 'error'}">${ok ? '已连接' : '异常'}</span>
  </div>`;
}

async function probeStorage() {
  const el = $('#storage-probe');
  el.innerHTML = '<div class="muted">正在连接…</div>';
  try {
    const d = await api('/api/storage/probe', { method: 'POST' });
    el.innerHTML = probeRow('Milvus', d.milvus) + probeRow('MinIO', d.minio);
  } catch (e) {
    el.textContent = '检测失败：' + e.message;
  }
}

async function resetStorage() {
  if (!confirm('清除已保存的地址，恢复 .env 中的默认值？')) return;
  try {
    fillStorage(await api('/api/storage/reset', { method: 'POST' }));
    $('#storage-status').textContent = '已恢复 .env 默认地址';
  } catch (e) {
    $('#storage-status').textContent = '恢复失败：' + e.message;
  }
}

refreshAll();