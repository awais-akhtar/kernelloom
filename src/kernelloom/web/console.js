(() => {
  "use strict";

  const byId = (id) => document.getElementById(id);
  const state = { models: [], collections: [], controller: null };

  const value = (id) => byId(id).value.trim();
  const numeric = (id) => Number(byId(id).value);
  const optionalNumeric = (id) => {
    const raw = value(id);
    return raw === "" ? undefined : Number(raw);
  };
  const checked = (id) => byId(id).checked;
  const stopStrings = (id) => byId(id).value.split(/\r?\n/).map((item) => item.trim()).filter(Boolean);

  function setStatus(id, message, kind = "") {
    const node = byId(id);
    node.textContent = message;
    node.className = `status ${kind}`.trim();
  }

  function parseObject(id, label) {
    const raw = value(id) || "{}";
    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch (error) {
      throw new Error(`${label} must be valid JSON.`);
    }
    if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") {
      throw new Error(`${label} must be a JSON object.`);
    }
    return parsed;
  }

  function apiHeaders() {
    const headers = { "Content-Type": "application/json" };
    const key = value("apiKey");
    if (key) headers.Authorization = `Bearer ${key}`;
    return headers;
  }

  async function api(path, options = {}) {
    const response = await fetch(path, { headers: apiHeaders(), ...options });
    const text = await response.text();
    let body = null;
    try { body = text ? JSON.parse(text) : null; } catch (_) { body = text; }
    if (!response.ok) {
      const detail = body && typeof body === "object" ? body.detail : text;
      throw new Error(detail || `${response.status} ${response.statusText}`);
    }
    return body;
  }

  function escapeFilename(value) {
    return value.replace(/[^a-z0-9._-]+/gi, "-").replace(/^-+|-+$/g, "") || "kernelloom";
  }

  function modelPayload() {
    const path = value("modelPath");
    if (!path) throw new Error("Model path is required.");
    return {
      model_path: path,
      model_id: value("modelId") || "default",
      backend: value("backend") || "auto",
      device: value("device") || "CPU",
      data_dir: value("dataDir"),
      context_length: numeric("contextLength"),
      batch_size: numeric("batchSize"),
      micro_batch_size: numeric("microBatchSize"),
      auto_batch_size: checked("autoBatchSize"),
      threads: numeric("threads"),
      batch_threads: numeric("batchThreads"),
      cpu_profile: value("cpuProfile") || "auto",
      reserve_cores: numeric("reserveCores"),
      gpu_layers: numeric("gpuLayers"),
      use_mmap: checked("useMmap"),
      use_mlock: checked("useMlock"),
      offload_kqv: checked("offloadKqv"),
      flash_attention: checked("flashAttention"),
      numa: checked("numa"),
      chat_format: value("chatFormat"),
      seed: numeric("seed"),
      embedding: checked("embedding"),
      embedding_cache_size: numeric("embeddingCacheSize"),
      embedding_cache_max_bytes: numeric("embeddingCacheBytes"),
      token_cache_size: numeric("tokenCacheSize"),
      warmup: checked("warmup"),
      warmup_prompt: value("warmupPrompt"),
      warmup_tokens: numeric("warmupTokens"),
      max_new_tokens: numeric("defaultMaxTokens"),
      temperature: numeric("defaultTemperature"),
      top_p: numeric("defaultTopP"),
      top_k: numeric("defaultTopK"),
      repetition_penalty: numeric("repetitionPenalty"),
      system_prompt: byId("systemPrompt").value,
      device_config: parseObject("deviceConfig", "Device configuration"),
      scheduler: parseObject("scheduler", "Scheduler configuration"),
    };
  }

  function assign(id, input) {
    if (input === undefined || input === null) return;
    const node = byId(id);
    if (node.type === "checkbox") node.checked = Boolean(input);
    else if (typeof input === "object") node.value = JSON.stringify(input, null, 2);
    else node.value = String(input);
  }

  function fillModelForm(config) {
    const fields = {
      modelPath: "model_path", modelId: "model_id", backend: "backend", device: "device", dataDir: "data_dir",
      contextLength: "context_length", batchSize: "batch_size", microBatchSize: "micro_batch_size",
      autoBatchSize: "auto_batch_size", threads: "threads", batchThreads: "batch_threads",
      cpuProfile: "cpu_profile", reserveCores: "reserve_cores", gpuLayers: "gpu_layers", useMmap: "use_mmap",
      useMlock: "use_mlock", offloadKqv: "offload_kqv", flashAttention: "flash_attention", numa: "numa",
      chatFormat: "chat_format", seed: "seed", embedding: "embedding", embeddingCacheSize: "embedding_cache_size",
      embeddingCacheBytes: "embedding_cache_max_bytes", tokenCacheSize: "token_cache_size", warmup: "warmup",
      warmupPrompt: "warmup_prompt", warmupTokens: "warmup_tokens", defaultMaxTokens: "max_new_tokens",
      defaultTemperature: "temperature", defaultTopP: "top_p", defaultTopK: "top_k",
      repetitionPenalty: "repetition_penalty", systemPrompt: "system_prompt", deviceConfig: "device_config", scheduler: "scheduler",
    };
    Object.entries(fields).forEach(([id, key]) => assign(id, config[key]));
    byId("modelForm").scrollIntoView({ behavior: "smooth", block: "start" });
    setStatus("modelStatus", `Loaded settings for ${config.model_id}.`, "ok");
  }

  function replaceOptions(id, models, selected, predicate = () => true, placeholder = "No eligible model loaded") {
    const select = byId(id);
    const current = selected || select.value;
    select.replaceChildren();
    const candidates = models.filter(predicate);
    if (!candidates.length) {
      const option = new Option(placeholder, "");
      option.disabled = true;
      option.selected = true;
      select.add(option);
      return;
    }
    candidates.forEach((model) => {
      const label = `${model.id} · ${model.backend} · ${model.device}${model.embedding ? " · embedding" : ""}`;
      select.add(new Option(label, model.id, false, model.id === current));
    });
    if (!select.value) select.selectedIndex = 0;
  }

  function modelButton(label, action, modelId, className = "secondary compact") {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.dataset.action = action;
    button.dataset.modelId = modelId;
    button.className = className;
    return button;
  }

  function renderModels() {
    const list = byId("modelCards");
    list.replaceChildren();
    byId("modelCount").textContent = `${state.models.length} model${state.models.length === 1 ? "" : "s"}`;
    if (!state.models.length) {
      const empty = document.createElement("p");
      empty.className = "muted";
      empty.textContent = "Load a model to start chatting, embedding, or creating a RAG collection.";
      list.append(empty);
    }
    state.models.forEach((model) => {
      const card = document.createElement("article");
      card.className = "model-card";
      const details = document.createElement("div");
      const title = document.createElement("div");
      title.className = "item-title";
      const name = document.createElement("strong");
      name.textContent = model.id;
      title.append(name);
      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = model.embedding ? "embedding" : `${model.backend} · ${model.device}`;
      title.append(badge);
      details.append(title);
      const path = document.createElement("p");
      path.className = "item-meta";
      const cache = model.cache || {};
      path.textContent = `${model.path} · warm: ${model.warmup?.warmed ? "yes" : "no"} · cache: ${cache.embedding_entries || 0} embeddings / ${cache.token_entries || 0} tokens`;
      details.append(path);
      const actions = document.createElement("div");
      actions.className = "item-actions";
      actions.append(
        modelButton("Use settings", "use", model.id),
        modelButton("Warm", "warm", model.id),
        modelButton("Clear cache", "cache", model.id),
        modelButton("Unload", "unload", model.id, "danger compact"),
      );
      card.append(details, actions);
      list.append(card);
    });

    const selectedChat = value("chatModel") || value("modelId");
    replaceOptions("chatModel", state.models, selectedChat);
    replaceOptions("ragModel", state.models, value("ragModel"), (model) => !model.embedding, "Load a chat model first");
    replaceOptions("ragEmbeddingModel", state.models, value("ragEmbeddingModel"), (model) => model.embedding, "Load an embedding model first");
  }

  async function refreshModels() {
    const result = await api("/v1/models");
    state.models = result.data || [];
    renderModels();
  }

  function collectionButton(label, action, id, className = "secondary compact") {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.dataset.collectionAction = action;
    button.dataset.collectionId = id;
    button.className = className;
    return button;
  }

  function renderCollections() {
    const list = byId("ragCollections");
    list.replaceChildren();
    if (!state.collections.length) {
      const empty = document.createElement("p");
      empty.className = "muted";
      empty.textContent = "No RAG collections are active.";
      list.append(empty);
    }
    state.collections.forEach((collection) => {
      const card = document.createElement("article");
      card.className = "collection-card";
      const details = document.createElement("div");
      const title = document.createElement("div");
      title.className = "item-title";
      const name = document.createElement("strong");
      name.textContent = collection.id;
      const badge = document.createElement("span");
      badge.className = collection.ready ? "badge" : "badge neutral";
      badge.textContent = collection.ready ? collection.store : "needs attention";
      title.append(name, badge);
      details.append(title);
      const meta = document.createElement("p");
      meta.className = "item-meta";
      const cache = collection.cache || {};
      meta.textContent = `${collection.documents} chunks · chat: ${collection.model} · embeddings: ${collection.embedding_model} · namespace: ${collection.config?.namespace || "default"} · query cache: ${cache.query_entries || 0}`;
      details.append(meta);
      if (!collection.ready && collection.reason) {
        const reason = document.createElement("p");
        reason.className = "item-meta";
        reason.textContent = collection.reason;
        details.append(reason);
      }
      const actions = document.createElement("div");
      actions.className = "item-actions";
      actions.append(
        collectionButton("Select", "select", collection.id),
        collectionButton("Warm", "warm", collection.id),
        collectionButton("Clear namespace", "clear", collection.id),
        collectionButton("Remove", "remove", collection.id, "danger compact"),
      );
      card.append(details, actions);
      list.append(card);
    });

    const current = value("ragQueryId") || value("ragIngestId");
    ["ragIngestId", "ragQueryId"].forEach((id) => {
      const select = byId(id);
      select.replaceChildren();
      if (!state.collections.length) {
        const option = new Option("Create a collection first", "");
        option.disabled = true;
        option.selected = true;
        select.add(option);
        return;
      }
      state.collections.forEach((collection) => select.add(new Option(
        `${collection.id} · ${collection.documents} chunks`, collection.id, false, collection.id === current,
      )));
      if (!select.value) select.selectedIndex = 0;
    });
  }

  async function refreshCollections() {
    const result = await api("/v1/rag/collections");
    state.collections = result.data || [];
    renderCollections();
  }

  async function refreshAll() {
    setStatus("modelStatus", "Refreshing…");
    try {
      await refreshModels();
      await refreshCollections();
      setStatus("modelStatus", "Runtime state refreshed.", "ok");
    } catch (error) {
      setStatus("modelStatus", error.message, "error");
    }
  }

  async function loadModel(event) {
    event.preventDefault();
    try {
      setStatus("modelStatus", "Loading model…");
      const model = await api("/v1/models/load", { method: "POST", body: JSON.stringify(modelPayload()) });
      setStatus("modelStatus", `${model.reused ? "Reused" : "Loaded"} ${model.id} on ${model.device}.`, "ok");
      await refreshModels();
    } catch (error) {
      setStatus("modelStatus", error.message, "error");
    }
  }

  async function modelAction(event) {
    const button = event.target.closest("button[data-action]");
    if (!button) return;
    const { action, modelId } = button.dataset;
    try {
      if (action === "use") {
        const detail = await api(`/v1/models/${encodeURIComponent(modelId)}`);
        fillModelForm(detail.config);
        return;
      }
      if (action === "warm") {
        await api(`/v1/models/${encodeURIComponent(modelId)}/warm`, { method: "POST", body: "{}" });
        setStatus("modelStatus", `${modelId} is warm.`, "ok");
      }
      if (action === "cache") {
        await api(`/v1/models/${encodeURIComponent(modelId)}/cache/clear`, { method: "POST", body: "{}" });
        setStatus("modelStatus", `Cleared ${modelId} caches.`, "ok");
      }
      if (action === "unload") {
        await api(`/v1/models/${encodeURIComponent(modelId)}`, { method: "DELETE" });
        setStatus("modelStatus", `Unloaded ${modelId}.`, "ok");
      }
      await refreshModels();
      await refreshCollections();
    } catch (error) {
      setStatus("modelStatus", error.message, "error");
    }
  }

  async function applyCpuPlan() {
    try {
      const profile = encodeURIComponent(value("cpuProfile") || "auto");
      const plan = await api(`/v1/cpu-plan?profile=${profile}&reserve_cores=${numeric("reserveCores")}`);
      assign("threads", plan.threads);
      assign("batchThreads", plan.batch_threads);
      assign("batchSize", plan.recommended_batch_size);
      assign("microBatchSize", plan.recommended_micro_batch_size);
      byId("autoBatchSize").checked = true;
      setStatus("modelStatus", plan.rationale, "ok");
    } catch (error) {
      setStatus("modelStatus", error.message, "error");
    }
  }

  async function inspectHardware() {
    try {
      byId("hardwareOutput").textContent = "Inspecting local hardware…";
      const profile = await api("/v1/hardware?refresh=true");
      byId("hardwareOutput").textContent = JSON.stringify(profile, null, 2);
    } catch (error) {
      byId("hardwareOutput").textContent = error.message;
    }
  }

  async function exportConfig() {
    try {
      const config = await api("/v1/runtime/config");
      const blob = new Blob([`${JSON.stringify(config, null, 2)}\n`], { type: "application/json" });
      const href = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = href;
      anchor.download = `${escapeFilename("kernelloom")}.json`;
      document.body.append(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(href);
      setStatus("modelStatus", "Downloaded a reusable runtime configuration.", "ok");
    } catch (error) {
      setStatus("modelStatus", error.message, "error");
    }
  }

  async function generateChat() {
    const model = value("chatModel");
    const prompt = byId("chatPrompt").value.trim();
    if (!model) return setStatus("chatStatus", "Load a model first.", "error");
    if (!prompt) return setStatus("chatStatus", "Enter a prompt.", "error");
    const payload = {
      model,
      messages: [{ role: "user", content: prompt }],
      max_tokens: numeric("chatTokens"),
      temperature: numeric("chatTemperature"),
      top_p: numeric("chatTopP"),
      top_k: numeric("chatTopK"),
      repetition_penalty: numeric("chatRepeatPenalty"),
    };
    const stops = stopStrings("chatStop");
    if (stops.length) payload.stop = stops;
    const answer = byId("chatAnswer");
    answer.textContent = "";
    setStatus("chatStatus", "Generating…");
    byId("sendChat").disabled = true;
    byId("stopChat").disabled = false;
    const started = performance.now();
    try {
      if (!checked("chatStream")) {
        const result = await api("/v1/chat/completions", { method: "POST", body: JSON.stringify(payload) });
        answer.textContent = result.choices?.[0]?.message?.content || "";
        const details = result.kernelloom || {};
        setStatus("chatStatus", `${Math.round(performance.now() - started)} ms · ${details.backend || "local"} · ${details.device || ""}`, "ok");
        return;
      }
      state.controller = new AbortController();
      const response = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: apiHeaders(),
        body: JSON.stringify({ ...payload, stream: true }),
        signal: state.controller.signal,
      });
      if (!response.ok || !response.body) {
        const text = await response.text();
        let message = text;
        try { message = JSON.parse(text).detail || text; } catch (_) { /* use raw text */ }
        throw new Error(message || "Streaming request failed.");
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value: chunk, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(chunk, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() || "";
        events.forEach((event) => {
          const data = event.split("\n").find((line) => line.startsWith("data: "))?.slice(6);
          if (!data || data === "[DONE]") return;
          try {
            const parsed = JSON.parse(data);
            if (parsed.error) throw new Error(parsed.error.message || "Streaming error");
            answer.textContent += parsed.choices?.[0]?.delta?.content || "";
          } catch (error) {
            if (error instanceof Error) throw error;
          }
        });
      }
      setStatus("chatStatus", `${Math.round(performance.now() - started)} ms · streamed locally`, "ok");
    } catch (error) {
      if (error.name === "AbortError") setStatus("chatStatus", "Generation stopped.", "warning");
      else setStatus("chatStatus", error.message, "error");
    } finally {
      state.controller = null;
      byId("sendChat").disabled = false;
      byId("stopChat").disabled = true;
    }
  }

  function ragCreatePayload() {
    const store = value("ragStore");
    const database = store === "sqlite" ? value("ragDatabase") : store;
    if (store === "sqlite" && !database) throw new Error("A SQLite path is required for the SQLite store.");
    return {
      id: value("ragId"),
      model: value("ragModel"),
      embedding_model: value("ragEmbeddingModel"),
      database,
      config: {
        namespace: value("ragNamespace") || "default",
        chunk_size: numeric("ragChunkSize"),
        chunk_overlap: numeric("ragChunkOverlap"),
        top_k: numeric("ragTopK"),
        fetch_k: numeric("ragFetchK"),
        max_context_chars: numeric("ragMaxContext"),
        min_score: numeric("ragMinScore"),
        retrieval: value("ragRetrieval"),
        mmr_lambda: numeric("ragMmrLambda"),
        include_sources: checked("ragIncludeSources"),
        query_cache_size: numeric("ragCacheSize"),
        query_cache_ttl_seconds: numeric("ragCacheTtl"),
        system_prompt: byId("ragSystemPrompt").value,
        prompt_template: byId("ragPromptTemplate").value,
      },
    };
  }

  async function createCollection(event) {
    event.preventDefault();
    try {
      setStatus("ragStatus", "Creating collection…");
      const collection = await api("/v1/rag/collections", { method: "POST", body: JSON.stringify(ragCreatePayload()) });
      setStatus("ragStatus", `Created ${collection.id}.`, "ok");
      await refreshCollections();
    } catch (error) {
      setStatus("ragStatus", error.message, "error");
    }
  }

  function sourcePayload() {
    const raw = byId("ragSources").value;
    if (!raw.trim()) throw new Error("Enter a server path or inline text.");
    return checked("ragSplitLines") ? raw.split(/\r?\n/).map((line) => line.trim()).filter(Boolean) : raw;
  }

  async function ingestCollection(event) {
    event.preventDefault();
    const id = value("ragIngestId");
    if (!id) return setStatus("ragStatus", "Create a collection first.", "error");
    try {
      setStatus("ragStatus", "Embedding and indexing…");
      const result = await api(`/v1/rag/collections/${encodeURIComponent(id)}/ingest`, {
        method: "POST",
        body: JSON.stringify({
          sources: sourcePayload(),
          namespace: value("ragIngestNamespace") || undefined,
          batch_size: numeric("ragBatchSize"),
          metadata: parseObject("ragMetadata", "Metadata"),
        }),
      });
      setStatus("ragStatus", `Indexed ${result.indexed} chunks in ${result.namespace}.`, "ok");
      await refreshCollections();
    } catch (error) {
      setStatus("ragStatus", error.message, "error");
    }
  }

  function ragQuestionPayload() {
    const question = byId("ragQuestion").value.trim();
    if (!question) throw new Error("Enter a question.");
    return {
      question,
      namespace: value("ragQueryNamespace") || undefined,
      top_k: optionalNumeric("ragQueryTopK"),
      filters: parseObject("ragFilters", "Filters"),
    };
  }

  function selectedCollection() {
    const id = value("ragQueryId");
    if (!id) throw new Error("Create a collection first.");
    return id;
  }

  async function retrieveCollection() {
    try {
      const id = selectedCollection();
      setStatus("ragStatus", "Retrieving…");
      const result = await api(`/v1/rag/collections/${encodeURIComponent(id)}/retrieve`, {
        method: "POST", body: JSON.stringify(ragQuestionPayload()),
      });
      const lines = (result.data || []).map((item, index) => {
        const source = item.metadata?.source || item.id;
        return `[${index + 1}] ${source} · score ${Number(item.score).toFixed(4)}\n${item.text}`;
      });
      byId("ragAnswer").textContent = lines.join("\n\n") || "No matching chunks found.";
      setStatus("ragStatus", `${result.data?.length || 0} chunks retrieved.`, "ok");
    } catch (error) {
      setStatus("ragStatus", error.message, "error");
    }
  }

  async function askCollection(event) {
    event.preventDefault();
    try {
      const id = selectedCollection();
      setStatus("ragStatus", "Retrieving and generating…");
      const generation = {
        max_new_tokens: numeric("ragQueryTokens"),
        temperature: numeric("ragQueryTemperature"),
        top_p: numeric("ragQueryTopP"),
        top_k: numeric("ragQueryGenerationTopK"),
        repetition_penalty: numeric("ragQueryRepeatPenalty"),
      };
      const stops = stopStrings("ragQueryStop");
      if (stops.length) generation.stop_strings = stops;
      const result = await api(`/v1/rag/collections/${encodeURIComponent(id)}/query`, {
        method: "POST",
        body: JSON.stringify({
          ...ragQuestionPayload(),
          generation,
        }),
      });
      const sources = (result.sources || []).map((item, index) => {
        const source = item.metadata?.source || item.id;
        return `[${index + 1}] ${source} · score ${Number(item.score).toFixed(4)}`;
      });
      byId("ragAnswer").textContent = `${result.answer || ""}${sources.length ? `\n\nRetrieved sources\n${sources.join("\n")}` : ""}`;
      setStatus("ragStatus", `Answered from ${sources.length} retrieved chunks.`, "ok");
      await refreshCollections();
    } catch (error) {
      setStatus("ragStatus", error.message, "error");
    }
  }

  async function warmCollection(id = value("ragQueryId")) {
    if (!id) return setStatus("ragStatus", "Create a collection first.", "error");
    try {
      setStatus("ragStatus", "Warming collection…");
      const question = byId("ragQuestion").value.trim();
      const result = await api(`/v1/rag/collections/${encodeURIComponent(id)}/warm`, {
        method: "POST", body: JSON.stringify({ queries: question ? [question] : [] }),
      });
      setStatus("ragStatus", `Warmup finished (${result.queries || 0} cached queries).`, "ok");
      await refreshCollections();
    } catch (error) {
      setStatus("ragStatus", error.message, "error");
    }
  }

  async function collectionAction(event) {
    const button = event.target.closest("button[data-collection-action]");
    if (!button) return;
    const { collectionAction: action, collectionId } = button.dataset;
    try {
      if (action === "select") {
        ["ragIngestId", "ragQueryId"].forEach((id) => { byId(id).value = collectionId; });
        setStatus("ragStatus", `Selected ${collectionId}.`, "ok");
        return;
      }
      if (action === "warm") {
        await warmCollection(collectionId);
        return;
      }
      if (action === "clear") {
        const collection = state.collections.find((item) => item.id === collectionId);
        const namespace = value("ragQueryNamespace") || collection?.config?.namespace || "default";
        const result = await api(`/v1/rag/collections/${encodeURIComponent(collectionId)}/namespaces/${encodeURIComponent(namespace)}`, { method: "DELETE" });
        setStatus("ragStatus", `Deleted ${result.deleted} chunks from ${namespace}.`, "ok");
      }
      if (action === "remove") {
        await api(`/v1/rag/collections/${encodeURIComponent(collectionId)}`, { method: "DELETE" });
        setStatus("ragStatus", `Removed ${collectionId}.`, "ok");
      }
      await refreshCollections();
    } catch (error) {
      setStatus("ragStatus", error.message, "error");
    }
  }

  byId("modelForm").addEventListener("submit", loadModel);
  byId("modelCards").addEventListener("click", modelAction);
  byId("planCpu").addEventListener("click", applyCpuPlan);
  byId("inspectHardware").addEventListener("click", inspectHardware);
  byId("exportConfig").addEventListener("click", exportConfig);
  byId("refreshAll").addEventListener("click", refreshAll);
  byId("sendChat").addEventListener("click", generateChat);
  byId("stopChat").addEventListener("click", () => state.controller?.abort());
  byId("ragCreate").addEventListener("submit", createCollection);
  byId("ragIngest").addEventListener("submit", ingestCollection);
  byId("ragQuery").addEventListener("submit", askCollection);
  byId("ragRetrieve").addEventListener("click", retrieveCollection);
  byId("ragWarm").addEventListener("click", () => warmCollection());
  byId("ragCollections").addEventListener("click", collectionAction);

  refreshAll();
})();
