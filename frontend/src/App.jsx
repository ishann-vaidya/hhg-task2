import React, { useState, useEffect, useRef } from "react";
import {
  Mic,
  MicOff,
  Upload,
  Clock,
  Shield,
  Activity,
  FileText,
  CheckCircle,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  Sliders,
  Settings,
  HelpCircle,
  CornerDownRight,
  Play,
  Square,
  AlertCircle
} from "lucide-react";

// Server URL - dynamically determine if we are running in the same origin or fallback
const API_BASE = "http://localhost:8000";

export default function App() {
  // Config state
  const [strategy, setStrategy] = useState("metadata_aware");
  const [threshold, setThreshold] = useState(0.42);
  const [isMock, setIsMock] = useState(true);
  const [apiStatus, setApiStatus] = useState({
    groq_configured: false,
    sarvam_configured: false,
    live_mode_ready: false
  });

  // Query & inputs state
  const [query, setQuery] = useState("");
  const [audioFile, setAudioFile] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioUrl, setAudioUrl] = useState("");
  const [audioBlob, setAudioBlob] = useState(null);

  // System run state
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [latencyReport, setLatencyReport] = useState(null);

  // Expandable UI toggles
  const [showConfig, setShowConfig] = useState(true);
  const [openCitations, setOpenCitations] = useState(false);
  const [openReasoning, setOpenReasoning] = useState(false);

  // Refs for audio recording
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);

  // Load API status and Latency reports on mount
  useEffect(() => {
    fetchApiStatus();
    fetchLatencyReport();
  }, []);

  // Update timer for recording
  useEffect(() => {
    if (isRecording) {
      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
      setRecordingTime(0);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isRecording]);

  const fetchApiStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/status`);
      const data = await res.json();
      setApiStatus(data);
      // Auto-configure mock mode based on key availability
      setIsMock(!data.live_mode_ready);
    } catch (err) {
      console.error("Failed to fetch API status", err);
    }
  };

  const fetchLatencyReport = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/latency`);
      const data = await res.json();
      if (data.status !== "warning") {
        setLatencyReport(data);
      }
    } catch (err) {
      console.error("Failed to load latency report", err);
    }
  };

  // HTML5 MediaRecorder voice capture
  const startRecording = async () => {
    audioChunksRef.current = [];
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        setAudioBlob(blob);
        setAudioUrl(URL.createObjectURL(blob));
        setAudioFile(null); // clear uploaded file
      };

      recorder.start(250); // capture chunks every 250ms
      setIsRecording(true);
    } catch (err) {
      alert("Error accessing microphone: Please grant microphone permissions.");
      console.error(err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      // stop all tracks
      mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop());
      setIsRecording(false);
    }
  };

  const resetInputs = () => {
    setQuery("");
    setAudioFile(null);
    setAudioBlob(null);
    setAudioUrl("");
    setResponse(null);
  };

  const executePipeline = async () => {
    if (!audioBlob && !audioFile && !query.trim()) {
      alert("Please enter a query text, select a sample prompt, or capture audio.");
      return;
    }

    setLoading(true);
    setResponse(null);
    const startTime = performance.now();

    try {
      let data;
      // 1. Audio Upload Route (WAV/MP3/WebM)
      if (audioBlob || audioFile) {
        const formData = new FormData();
        if (audioBlob) {
          formData.append("file", audioBlob, "web_recording.webm");
        } else {
          formData.append("file", audioFile);
        }
        formData.append("strategy", strategy);
        formData.append("threshold", threshold.toString());
        formData.append("mock", isMock.toString());
        formData.append("mock_text", query || "निगम क्या है?");

        const res = await fetch(`${API_BASE}/api/predict/audio`, {
          method: "POST",
          body: formData
        });
        data = await res.json();
      }
      // 2. Text Bypass Route
      else {
        const res = await fetch(`${API_BASE}/api/predict/text`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: query,
            strategy: strategy,
            threshold: threshold,
            mock: isMock
          })
        });
        data = await res.json();
      }

      setResponse(data);
      // Reload latency report after runs
      fetchLatencyReport();
    } catch (err) {
      console.error(err);
      alert("Failed to process request. Verify your FastAPI backend is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  // Quick preset loader
  const loadPreset = (text) => {
    setQuery(text);
    setAudioFile(null);
    setAudioBlob(null);
    setAudioUrl("");
  };

  const presets = [
    { title: "🏢 निगम परिभाषा", text: "कॉर्पोरेशन क्या है?" },
    { title: "🥔 पोटेशियम सूची", text: "पोटेशियम में कम खाद्य पदार्थों का चार्ट।" },
    { title: "📖 दायित्व बर्दाश्त", text: "रेचल कार्सन ने क्यों एक दायित्व बर्दाश्त करने के लिए लिखा" }
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col antialiased">
      {/* ── Top Header Bar ── */}
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur-md px-6 py-4 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="flex items-center gap-3">
            <div className="bg-indigo-600 p-2.5 rounded-xl shadow-lg shadow-indigo-600/20">
              <Mic className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-indigo-200 to-indigo-400 bg-clip-text text-transparent">
                Indic Voice Console
              </h1>
              <p className="text-xs text-slate-400">Decoupled Multilingual Voice-RAG (HH Goa 2026)</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
              <span className="text-xs text-slate-300 font-medium bg-slate-800/80 px-2.5 py-1 rounded-full border border-slate-700">
                Backend Status: Connected
              </span>
            </div>

            <div className="flex items-center gap-2">
              {apiStatus.live_mode_ready ? (
                <span className="text-xs font-semibold bg-emerald-950/80 text-emerald-400 border border-emerald-800/50 px-3 py-1 rounded-full">
                  🟢 Live API Mode (Groq/Sarvam)
                </span>
              ) : (
                <span className="text-xs font-semibold bg-amber-950/80 text-amber-400 border border-amber-800/50 px-3 py-1 rounded-full">
                  🟡 Demo Mode (Simulated APIs)
                </span>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* ── Main Layout Workspace ── */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* ── Left Column: Config Panel ── */}
        <section className="lg:col-span-4 flex flex-col gap-6">
          
          {/* Strategy & Threshold Controls Card */}
          <div className="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-5 backdrop-blur-md">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
                <Sliders className="h-4 w-4 text-indigo-400" />
                Parameters
              </h2>
              <button 
                onClick={() => setShowConfig(!showConfig)}
                className="text-slate-400 hover:text-slate-200"
              >
                {showConfig ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
              </button>
            </div>

            {showConfig && (
              <div className="flex flex-col gap-4">
                <div>
                  <label className="text-xs text-slate-400 block mb-1.5 font-medium">FAISS Chunk Strategy</label>
                  <select
                    value={strategy}
                    onChange={(e) => setStrategy(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg py-2 px-3 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
                  >
                    <option value="metadata_aware">Metadata-Aware (Paragraph first)</option>
                    <option value="semantic">Semantic (Similarity splits)</option>
                    <option value="fixed_overlap">Fixed Size + Overlap (256/50)</option>
                    <option value="fixed_size">Fixed Size Baseline (256 tok)</option>
                  </select>
                </div>

                <div>
                  <div className="flex justify-between text-xs text-slate-400 mb-1.5 font-medium">
                    <span>Off-Topic Score Cutoff</span>
                    <span className="text-indigo-400 font-semibold">{threshold.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.20"
                    max="0.80"
                    step="0.01"
                    value={threshold}
                    onChange={(e) => setThreshold(parseFloat(e.target.value))}
                    className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                  />
                  <span className="text-[10px] text-slate-500 block mt-1">Queries scoring below this similarity will be blocked as off-topic.</span>
                </div>

                <div className="border-t border-slate-800/80 pt-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="text-xs font-semibold text-slate-300 block">Force Mock Mode</span>
                      <span className="text-[10px] text-slate-500">Run local simulator with preset latencies.</span>
                    </div>
                    <input
                      type="checkbox"
                      checked={isMock}
                      onChange={(e) => setIsMock(e.target.checked)}
                      disabled={!apiStatus.live_mode_ready}
                      className="h-4.5 w-4.5 rounded border-slate-800 text-indigo-600 focus:ring-indigo-500 accent-indigo-500 cursor-pointer"
                    />
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Latency Percentiles Analytics Board */}
          <div className="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-5 backdrop-blur-md">
            <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-4 flex items-center gap-2">
              <Activity className="h-4 w-4 text-indigo-400" />
              Latency analytics (P50/P70/P100)
            </h2>
            {latencyReport ? (
              <div className="flex flex-col gap-3">
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-900 flex justify-between items-center">
                  <span className="text-xs text-slate-400">STT Time (Mock)</span>
                  <div className="text-xs text-slate-300 font-semibold flex gap-2">
                    <span>P50: <b className="text-indigo-300">{latencyReport.summary.stt?.p50}ms</b></span>
                    <span>P99: <b className="text-indigo-400">{latencyReport.summary.stt?.p100}ms</b></span>
                  </div>
                </div>
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-900 flex justify-between items-center">
                  <span className="text-xs text-slate-400">Retrieval (FAISS)</span>
                  <div className="text-xs text-slate-300 font-semibold flex gap-2">
                    <span>P50: <b className="text-indigo-300">{latencyReport.summary.retrieval?.p50}ms</b></span>
                    <span>P99: <b className="text-indigo-400">{latencyReport.summary.retrieval?.p100}ms</b></span>
                  </div>
                </div>
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-900 flex justify-between items-center">
                  <span className="text-xs text-slate-400">LLM Generation</span>
                  <div className="text-xs text-slate-300 font-semibold flex gap-2">
                    <span>P50: <b className="text-indigo-300">{latencyReport.summary.generation?.p50}ms</b></span>
                    <span>P99: <b className="text-indigo-400">{latencyReport.summary.generation?.p100}ms</b></span>
                  </div>
                </div>
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-900 flex justify-between items-center">
                  <span className="text-xs text-slate-400 font-medium">Total pipeline E2E</span>
                  <div className="text-xs text-slate-200 font-bold flex gap-2">
                    <span>P50: <b className="text-indigo-400">{latencyReport.summary.total?.p50}ms</b></span>
                    <span>P99: <b className="text-indigo-500">{latencyReport.summary.total?.p100}ms</b></span>
                  </div>
                </div>
                <span className="text-[10px] text-slate-500 block text-center mt-1">Calculated across {latencyReport.config.n} validation queries.</span>
              </div>
            ) : (
              <div className="text-center py-6 text-slate-500 text-xs flex flex-col items-center gap-2">
                <AlertCircle className="h-6 w-6 text-slate-600" />
                No latency report found. Please run the phase7 benchmarking script to populate.
              </div>
            )}
          </div>
        </section>

        {/* ── Right Column: Interactive Console & Results ── */}
        <section className="lg:col-span-8 flex flex-col gap-6">
          
          {/* Voice Input Dashboard Console */}
          <div className="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-6 backdrop-blur-md">
            <h2 className="text-md font-semibold text-slate-200 mb-4">Input Console</h2>
            
            <div className="flex flex-col items-center justify-center py-6 bg-slate-950/40 rounded-2xl border border-slate-900/60 mb-6">
              
              {/* Pulsing Mic Aura */}
              <div className="relative mb-4">
                <button
                  onClick={isRecording ? stopRecording : startRecording}
                  disabled={loading}
                  className={`w-20 h-20 rounded-full flex items-center justify-center transition-all ${
                    isRecording 
                      ? "bg-red-600 shadow-red-500/50 shadow-lg scale-105" 
                      : "bg-indigo-600 hover:bg-indigo-500 shadow-indigo-500/35 shadow-lg hover:scale-105"
                  } disabled:opacity-50`}
                >
                  {isRecording ? (
                    <MicOff className="h-8 w-8 text-white" />
                  ) : (
                    <Mic className="h-8 w-8 text-white" />
                  )}
                </button>
                {isRecording && (
                  <span className="absolute -inset-2 rounded-full border border-red-500/40 animate-ping z-0 pointer-events-none"></span>
                )}
              </div>

              <div className="text-center">
                <p className="text-sm font-semibold">
                  {isRecording ? `Recording... (${recordingTime}s)` : "Press microphone to record speech"}
                </p>
                <p className="text-xs text-slate-500 mt-1">Accepts voice queries in Hindi. Transcribes via Sarvam AI.</p>
              </div>

              {/* Recorded Audio Playback */}
              {audioUrl && !isRecording && (
                <div className="mt-4 flex items-center gap-3 bg-slate-900/80 p-2.5 rounded-lg border border-slate-800">
                  <span className="text-xs text-slate-400">Microphone capture:</span>
                  <audio src={audioUrl} controls className="h-7 w-52 max-w-xs scale-90" />
                </div>
              )}
            </div>

            {/* Upload File Option */}
            <div className="flex flex-col md:flex-row gap-4 mb-4">
              <div className="flex-1">
                <label className="text-xs text-slate-400 block mb-1 font-medium">Or upload spoken audio file</label>
                <div className="flex items-center bg-slate-950 border border-slate-900 rounded-lg p-1">
                  <input
                    type="file"
                    accept="audio/wav, audio/mp3, audio/webm"
                    onChange={(e) => {
                      if (e.target.files.length > 0) {
                        setAudioFile(e.target.files[0]);
                        setAudioBlob(null);
                        setAudioUrl("");
                      }
                    }}
                    className="block w-full text-xs text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-slate-900 file:text-indigo-400 hover:file:bg-slate-850 cursor-pointer"
                  />
                </div>
              </div>

              <div className="flex-1">
                <label className="text-xs text-slate-400 block mb-1 font-medium">Or type query text directly</label>
                <input
                  type="text"
                  placeholder="Type Hindi question here..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-900 rounded-lg py-2.5 px-3 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-indigo-500"
                />
              </div>
            </div>

            {/* Quick click tags */}
            <div className="flex flex-wrap gap-2 items-center mb-6">
              <span className="text-[10px] text-slate-500 font-medium">Preset Prompts:</span>
              {presets.map((p, idx) => (
                <button
                  key={idx}
                  onClick={() => loadPreset(p.text)}
                  className="text-[11px] bg-indigo-950/40 text-indigo-300 hover:bg-indigo-950/80 border border-indigo-900/60 px-2.5 py-1 rounded-md transition-colors"
                >
                  {p.title}
                </button>
              ))}
            </div>

            {/* Primary Action Buttons */}
            <div className="flex gap-4">
              <button
                onClick={executePipeline}
                disabled={loading}
                className="flex-1 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-semibold py-2.5 px-4 rounded-lg shadow-lg shadow-indigo-600/10 hover:shadow-indigo-600/25 transition-all flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    Executing Pipeline...
                  </>
                ) : (
                  "🚀 Execute pipeline"
                )}
              </button>

              <button
                onClick={resetInputs}
                className="bg-slate-900 hover:bg-slate-850 border border-slate-800 text-slate-400 hover:text-slate-200 text-sm font-medium py-2.5 px-4 rounded-lg transition-all"
              >
                Reset
              </button>
            </div>
          </div>

          {/* ── System Response Cards ── */}
          {response && (
            <div className="flex flex-col gap-6 animate-fadeIn">
              
              {/* Output & Transcription Card */}
              <div className="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-6 backdrop-blur-md">
                <h3 className="text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                  <FileText className="h-4 w-4" />
                  Processed Output
                </h3>
                
                {/* Transcription display */}
                <div className="mb-4">
                  <span className="text-[10px] text-slate-500 block mb-1">STT Transcription query:</span>
                  <p className="text-md font-medium text-slate-200 bg-slate-950/60 p-3 rounded-lg border border-slate-900/60 italic">
                    "{response.query}"
                  </p>
                </div>

                {/* Final Generation response */}
                <div>
                  <span className="text-[10px] text-slate-500 block mb-1">Synthesized Answer:</span>
                  <div className={`p-4 rounded-xl border leading-relaxed text-sm font-medium ${
                    response.status.includes("blocked")
                      ? "bg-red-950/30 text-red-400 border-red-900/40"
                      : "bg-slate-950/60 text-slate-100 border-slate-900/60"
                  }`}>
                    {response.answer}
                  </div>
                </div>

                {/* Citations display */}
                {response.citations && response.citations.length > 0 && (
                  <div className="mt-4 pt-4 border-t border-slate-800/60">
                    <button
                      onClick={() => setOpenCitations(!openCitations)}
                      className="text-xs text-indigo-400 flex items-center gap-1 font-semibold hover:text-indigo-300"
                    >
                      References Used ({response.citations.length})
                      {openCitations ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                    </button>
                    {openCitations && (
                      <div className="mt-2 flex flex-col gap-1.5">
                        {response.citations.map((c, i) => (
                          <div key={i} className="text-xs text-slate-400 flex items-center gap-2">
                            <CornerDownRight className="h-3.5 w-3.5 text-slate-600" />
                            <span>Passage ID: <code className="text-indigo-300 bg-slate-950 px-1 rounded">{c}</code></span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Reasoning path */}
                {response.reasoning && (
                  <div className="mt-2">
                    <button
                      onClick={() => setOpenReasoning(!openReasoning)}
                      className="text-xs text-indigo-400 flex items-center gap-1 font-semibold hover:text-indigo-300"
                    >
                      Show reasoning trace
                      {openReasoning ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                    </button>
                    {openReasoning && (
                      <div className="mt-2 p-3 bg-slate-950/80 rounded-lg border border-slate-900 text-xs text-slate-400 font-mono leading-relaxed whitespace-pre-line">
                        {response.reasoning}
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Execution Flow & Guardrail Timeline Panel */}
              <div className="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-6 backdrop-blur-md grid grid-cols-1 md:grid-cols-2 gap-6">
                
                {/* Latency break down */}
                <div>
                  <h3 className="text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-4 flex items-center gap-1.5">
                    <Clock className="h-4 w-4" />
                    Latency trace
                  </h3>
                  <div className="flex flex-col gap-2">
                    <div className="flex justify-between items-center bg-slate-950/40 p-2.5 rounded-lg border border-slate-900/60">
                      <span className="text-xs text-slate-400">Speech-To-Text</span>
                      <span className="text-xs text-slate-300 font-semibold">{response.latencies.stt?.toFixed(1)} ms</span>
                    </div>
                    <div className="flex justify-between items-center bg-slate-950/40 p-2.5 rounded-lg border border-slate-900/60">
                      <span className="text-xs text-slate-400">Safety Check</span>
                      <span className="text-xs text-slate-300 font-semibold">{response.latencies.safety_guard?.toFixed(1)} ms</span>
                    </div>
                    <div className="flex justify-between items-center bg-slate-950/40 p-2.5 rounded-lg border border-slate-900/60">
                      <span className="text-xs text-slate-400">FAISS retrieval</span>
                      <span className="text-xs text-slate-300 font-semibold">{response.latencies.retrieval?.toFixed(1)} ms</span>
                    </div>
                    <div className="flex justify-between items-center bg-slate-950/40 p-2.5 rounded-lg border border-slate-900/60">
                      <span className="text-xs text-slate-400">LLM Generation</span>
                      <span className="text-xs text-slate-300 font-semibold">{response.latencies.generation?.toFixed(1)} ms</span>
                    </div>
                    <div className="flex justify-between items-center bg-slate-950/60 p-2.5 rounded-lg border border-indigo-900/40">
                      <span className="text-xs text-slate-200 font-semibold">Total pipeline latency</span>
                      <span className="text-xs text-indigo-400 font-bold">{response.latencies.total?.toFixed(1)} ms</span>
                    </div>
                  </div>
                </div>

                {/* Guardrails Check list */}
                <div>
                  <h3 className="text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-4 flex items-center gap-1.5">
                    <Shield className="h-4 w-4" />
                    Guardrail pipeline
                  </h3>
                  <div className="flex flex-col gap-2.5">
                    
                    {/* Safety */}
                    <div className="flex justify-between items-center bg-slate-950/40 p-2.5 rounded-lg border border-slate-900/60">
                      <span className="text-xs text-slate-300">🛡️ Input Safety</span>
                      {response.guardrails.safety.safe ? (
                        <span className="text-[10px] bg-emerald-950 text-emerald-400 px-2 py-0.5 rounded font-bold border border-emerald-900">Passed</span>
                      ) : (
                        <span className="text-[10px] bg-red-950 text-red-400 px-2 py-0.5 rounded font-bold border border-red-900">Violation</span>
                      )}
                    </div>

                    {/* Off-Topic */}
                    <div className="flex justify-between items-center bg-slate-950/40 p-2.5 rounded-lg border border-slate-900/60">
                      <span className="text-xs text-slate-300">🔍 Topic Scope</span>
                      {!response.guardrails.off_topic ? (
                        <span className="text-[10px] bg-slate-900 text-slate-500 px-2 py-0.5 rounded font-bold">Skipped</span>
                      ) : response.guardrails.off_topic.on_topic ? (
                        <span className="text-[10px] bg-emerald-950 text-emerald-400 px-2 py-0.5 rounded font-bold border border-emerald-900">Passed</span>
                      ) : (
                        <span className="text-[10px] bg-red-950 text-red-400 px-2 py-0.5 rounded font-bold border border-red-900">Off-topic</span>
                      )}
                    </div>

                    {/* Groundedness */}
                    <div className="flex justify-between items-center bg-slate-950/40 p-2.5 rounded-lg border border-slate-900/60">
                      <span className="text-xs text-slate-300">⚖️ Grounded Context</span>
                      {!response.guardrails.groundedness ? (
                        <span className="text-[10px] bg-slate-900 text-slate-500 px-2 py-0.5 rounded font-bold">Skipped</span>
                      ) : response.guardrails.groundedness.grounded ? (
                        <span className="text-[10px] bg-emerald-950 text-emerald-400 px-2 py-0.5 rounded font-bold border border-emerald-900">Grounded</span>
                      ) : (
                        <span className="text-[10px] bg-red-950 text-red-400 px-2 py-0.5 rounded font-bold border border-red-900">Hallucination</span>
                      )}
                    </div>
                  </div>
                  {/* Block explanation message if any */}
                  {response.status !== "success" && (
                    <div className="mt-3 text-xs bg-red-950/20 text-red-400/80 p-2 rounded-lg border border-red-900/30 flex items-start gap-2">
                      <AlertTriangle className="h-4 w-4 flex-shrink-0 mt-0.5" />
                      <span>{response.answer}</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Context passages */}
              <div className="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-6 backdrop-blur-md">
                <h3 className="text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-4">
                  📚 Retrieved Context Chunks from FAISS
                </h3>
                <div className="flex flex-col gap-4">
                  {response.chunks && response.chunks.length > 0 ? (
                    response.chunks.map((c, i) => (
                      <div key={i} className="bg-slate-950/60 p-4 rounded-xl border border-slate-900/60">
                        <div className="flex justify-between text-xs mb-2">
                          <span className="text-indigo-400 font-semibold">Chunk [{i}] ID: <code className="bg-slate-900 px-1 rounded">{c.passage_id}</code></span>
                          <span className="text-emerald-400 font-semibold">Similarity: {c.similarity_score.toFixed(4)}</span>
                        </div>
                        <p className="text-xs text-slate-300 leading-relaxed font-light">{c.text}</p>
                      </div>
                    ))
                  ) : (
                    <div className="text-center text-xs text-slate-500 py-4 italic">No context retrieved (Query blocked by safety checks).</div>
                  )}
                </div>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
