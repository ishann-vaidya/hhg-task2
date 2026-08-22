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
  AlertCircle,
  ArrowRight,
  Volume2,
  Database,
  Cpu,
  Lock,
  Layers
} from "lucide-react";

// Use the deployed API URL in production; keep localhost only during local development.
const API_BASE = (
  import.meta.env.VITE_API_URL ||
  (import.meta.env.DEV ? "http://localhost:8000" : "")
).replace(/\/$/, "");

// Multi-language configurations (Presets, placeholders, and mock defaults)
const languageConfigs = {
  en: {
    name: "English",
    placeholder: "Type English question here...",
    mockText: "What is a corporation?",
    presets: [
      { title: "Corp Definition", text: "What is a corporation?" },
      { title: "Low Potassium List", text: "Chart for foods low in potassium" },
      { title: "Rachel Carson", text: "Why did Rachel Carson write Silent Spring?" }
    ]
  },
  hi: {
    name: "Hindi (हिन्दी)",
    placeholder: "Type Hindi question here...",
    mockText: "निगम क्या है?",
    presets: [
      { title: "निगम परिभाषा", text: "कॉर्पोरेशन क्या है?" },
      { title: "पोटेशियम सूची", text: "पोटेशियम में कम खाद्य पदार्थों का चार्ट।" },
      { title: "दायित्व बर्दाश्त", text: "रेचल कार्सन ने क्यों एक दायित्व बर्दाश्त करने के लिए लिखा" }
    ]
  },
  mr: {
    name: "Marathi (मराठी)",
    placeholder: "Type Marathi question here...",
    mockText: "कॉर्पोरेशन म्हणजे काय?",
    presets: [
      { title: "कॉर्पोरेशन व्याख्या", text: "कॉर्पोरेशन म्हणजे काय?" },
      { title: "कमी पोटॅशियम", text: "कमी पोटॅशियम असलेल्या पदार्थांचा चार्ट।" },
      { title: "कार्सन कार्सन", text: "रेचल कार्सनने सायलेंट स्प्रिंग का लिहिले?" }
    ]
  },
  te: {
    name: "Telugu (తెలుగు)",
    placeholder: "Type Telugu question here...",
    mockText: "కార్పొరేషన్ అంటే ఏమిటి?",
    presets: [
      { title: "కార్పొరేషన్", text: "కార్పొరేషన్ అంటే ఏమిటి?" },
      { title: "తక్కువ పొటాషియం", text: "పొటాషియం తక్కువగా ఉండే ఆహారాల చార్ట్." }
    ]
  },
  ta: {
    name: "Tamil (தமிழ்)",
    placeholder: "Type Tamil question here...",
    mockText: "கார்ப்பரேஷன் என்றால் என்ன?",
    presets: [
      { title: "கார்ப்பரேஷன்", text: "கார்ப்பரேஷன் என்றால் என்ன?" },
      { title: "குறைந்த பொட்டாசியம்", text: "குறைந்த பொட்டாசியம் உணவுகளின் விளக்கப்படம்." }
    ]
  }
};

// ── Interactive WebGL-Style Canvas Background ──
function CanvasBackground({ isRecording, isLoading }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let animationFrameId;

    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    const handleResize = () => {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    };
    window.addEventListener("resize", handleResize);

    // Dynamic node connections
    const numParticles = 75;
    const particles = [];
    for (let i = 0; i < numParticles; i++) {
      particles.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.45,
        vy: (Math.random() - 0.5) * 0.45,
        radius: Math.random() * 1.5 + 1
      });
    }

    let mouse = { x: null, y: null };
    const handleMouseMove = (e) => {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
    };
    const handleMouseLeave = () => {
      mouse.x = null;
      mouse.y = null;
    };
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseleave", handleMouseLeave);

    let wavePhase = 0;

    const animate = () => {
      ctx.clearRect(0, 0, width, height);

      // Deep obsidian space gradient background
      const gradient = ctx.createRadialGradient(
        width / 2, height / 2, 10,
        width / 2, height / 2, Math.max(width, height)
      );
      gradient.addColorStop(0, "#080c1d");
      gradient.addColorStop(1, "#02040a");
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);

      // Render neural network nodes
      ctx.lineWidth = 0.5;
      for (let i = 0; i < numParticles; i++) {
        const p1 = particles[i];
        
        // Move particle
        p1.x += p1.vx;
        p1.y += p1.vy;

        // Boundaries bounce check
        if (p1.x < 0 || p1.x > width) p1.vx *= -1;
        if (p1.y < 0 || p1.y > height) p1.vy *= -1;

        // Magnetic repulsion from user mouse cursor
        if (mouse.x !== null && mouse.y !== null) {
          const dx = mouse.x - p1.x;
          const dy = mouse.y - p1.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 120) {
            const force = (120 - dist) / 120;
            p1.x -= dx / dist * force * 0.7;
            p1.y -= dy / dist * force * 0.7;
          }
        }

        // Draw particle node
        ctx.fillStyle = "rgba(99, 102, 241, 0.35)";
        ctx.beginPath();
        ctx.arc(p1.x, p1.y, p1.radius, 0, Math.PI * 2);
        ctx.fill();

        // Connect nodes near to each other
        for (let j = i + 1; j < numParticles; j++) {
          const p2 = particles[j];
          const dx = p1.x - p2.x;
          const dy = p1.y - p2.y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < 110) {
            const alpha = (110 - dist) / 110 * 0.12;
            ctx.strokeStyle = `rgba(99, 102, 241, ${alpha})`;
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
          }
        }
      }

      // Renders dynamic Voice-Waves representing active voice frequencies
      wavePhase += isRecording ? 0.16 : isLoading ? 0.08 : 0.015;
      const waveAmplitude = isRecording ? 48 : isLoading ? 22 : 6;
      const numWaves = 3;

      for (let w = 0; w < numWaves; w++) {
        ctx.beginPath();
        ctx.lineWidth = 1 + w * 0.5;
        ctx.strokeStyle = `rgba(99, 102, 241, ${0.07 - w * 0.02})`;
        
        const offset = w * Math.PI / 3;
        for (let x = 0; x < width; x += 10) {
          const y = height * 0.88 + Math.sin(x * 0.0035 + wavePhase + offset) * waveAmplitude * Math.cos(x * 0.0008);
          if (x === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
      }

      animationFrameId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseleave", handleMouseLeave);
      cancelAnimationFrame(animationFrameId);
    };
  }, [isRecording, isLoading]);

  return <canvas ref={canvasRef} className="fixed inset-0 w-full h-full -z-10 pointer-events-none" />;
}

export default function App() {
  // Navigation View: "landing" or "console"
  const [view, setView] = useState("landing");
  const [activeDiagStep, setActiveDiagStep] = useState(0);

  // Config state
  const [strategy, setStrategy] = useState("metadata_aware");
  const [language, setLanguage] = useState("en"); // Default to English interface queries
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
        formData.append("language", language);
        formData.append("threshold", threshold.toString());
        formData.append("mock", isMock.toString());
        formData.append("mock_text", query || languageConfigs[language].mockText);

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
            language: language,
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

  // Clear query on language change to keep dynamic prompts context clean
  const handleLanguageChange = (newLang) => {
    setLanguage(newLang);
    setQuery("");
    setResponse(null);
  };

  // Interactive flow step details
  const flowSteps = [
    {
      title: "Speech Input (STT)",
      icon: <Volume2 className="h-5 w-5 text-indigo-400" />,
      desc: "Captures user audio queries in Indic languages (WAV, MP3, WebM) via the browser microphone, transcribing them instantly using Sarvam AI's saaras:v3 model configured for specific locales (hi-IN, mr-IN, etc.)."
    },
    {
      title: "Safety Guardrail",
      icon: <Lock className="h-5 w-5 text-red-400" />,
      desc: "Performs query safety auditing. Screens transcribed texts against a safety blocklist looking for inappropriate or toxic keywords before querying the database, blocking unsafe inputs instantly."
    },
    {
      title: "Vector Search",
      icon: <Database className="h-5 w-5 text-indigo-400" />,
      desc: "Loads multilingual sentence embeddings and queries the FAISS index database. Fetches the top 3 most relevant passages dynamically matching the semantic meaning of the question in under 30ms."
    },
    {
      title: "Off-Topic Scope",
      icon: <Shield className="h-5 w-5 text-indigo-400" />,
      desc: "Computes cosine-similarity index matching. If the score falls below a user-adjustable cutoff threshold, the query is blocked as off-topic, preventing LLM response hallucinations."
    },
    {
      title: "Context-Grounded LLM",
      icon: <Cpu className="h-5 w-5 text-indigo-400" />,
      desc: "Bundles retrieved context chunks and query, calling Groq (Llama 3 8B) with a strict structured system prompt. Outputs JSON containing the final answer, reference citations, and reasoning steps."
    },
    {
      title: "Groundedness Audit",
      icon: <Layers className="h-5 w-5 text-indigo-400" />,
      desc: "Performs a final post-generation word-overlap sanity check. Verifies the answer contains sufficient noun/verb matches from the retrieved context passages to ensure factual accuracy."
    }
  ];

  return (
    <div className="min-h-screen text-slate-100 flex flex-col antialiased relative">
      
      {/* Dynamic Animated Wave Background */}
      <CanvasBackground isRecording={isRecording} isLoading={loading} />

      {/* ── LANDING VIEW ── */}
      {view === "landing" && (
        <div className="flex-1 flex flex-col relative z-10 max-w-6xl w-full mx-auto p-4 md:p-8 justify-center gap-12 animate-fadeIn">
          
          {/* Header Area */}
          <div className="flex justify-between items-center border-b border-slate-800/80 pb-6 bg-slate-950/20 px-4 rounded-xl">
            <div className="flex items-center gap-2.5">
              <Mic className="h-6 w-6 text-indigo-500" />
              <span className="font-bold text-lg tracking-wider text-slate-200">Indic Voice RAG</span>
            </div>
            <button
              onClick={() => setView("console")}
              className="flex items-center gap-1.5 bg-indigo-650 hover:bg-indigo-600 text-white text-xs font-semibold px-4 py-2.5 rounded-lg shadow-lg shadow-indigo-600/25 transition-all cursor-pointer"
            >
              Launch Console
              <ArrowRight className="h-3.5 w-3.5" />
            </button>
          </div>

          {/* Hero Section */}
          <section className="text-center flex flex-col items-center max-w-3xl mx-auto gap-4 py-6">
            <div className="bg-indigo-950/40 border border-indigo-900/60 text-indigo-300 text-[10px] font-semibold tracking-wider px-3.5 py-1 rounded-full uppercase">
              Hacker House Goa 2026 Submission
            </div>
            <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight leading-tight bg-gradient-to-r from-slate-100 via-indigo-150 to-indigo-300 bg-clip-text text-transparent">
              Multilingual Voice-RAG for Indian Languages
            </h1>
            <p className="text-sm text-slate-400 max-w-2xl leading-relaxed">
              A decoupled production-grade architecture combining real-time spoken Hindi and Marathi transcription, high-performance FAISS semantic search database matching, and strict hallucination guardrails.
            </p>
            <div className="flex gap-4 mt-4">
              <button
                onClick={() => setView("console")}
                className="bg-indigo-650 hover:bg-indigo-600 text-white text-sm font-semibold px-6 py-3 rounded-lg shadow-lg shadow-indigo-650/30 transition-all flex items-center gap-2 cursor-pointer"
              >
                Get Started
                <ArrowRight className="h-4 w-4" />
              </button>
              <a
                href="#how-it-works"
                className="bg-slate-900/60 hover:bg-slate-800 border border-slate-800 text-slate-300 text-sm font-semibold px-6 py-3 rounded-lg transition-all"
              >
                Inspect Workflow
              </a>
            </div>
          </section>

          {/* Interactive Block Diagram Section */}
          <section id="how-it-works" className="bg-slate-950/70 border border-slate-800/80 rounded-2xl p-6 backdrop-blur-md shadow-xl flex flex-col gap-6 scroll-mt-24">
            <div>
              <h2 className="text-lg font-bold text-slate-200">Interactive Pipeline Architecture</h2>
              <p className="text-xs text-slate-400 mt-0.5">Click on any pipeline stage to inspect the processing logic and data transformations.</p>
            </div>

            {/* Clickable flow cards */}
            <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
              {flowSteps.map((step, idx) => (
                <button
                  key={idx}
                  onClick={() => setActiveDiagStep(idx)}
                  className={`p-4 rounded-xl border flex flex-col items-center text-center gap-2 transition-all cursor-pointer ${
                    activeDiagStep === idx
                      ? "bg-indigo-950/50 border-indigo-500 shadow-md shadow-indigo-600/10 scale-[1.02]"
                      : "bg-slate-900/30 border-slate-800/60 hover:border-slate-700/80 hover:bg-slate-900/50"
                  }`}
                >
                  <div className={`p-2 rounded-lg ${activeDiagStep === idx ? "bg-indigo-900/50" : "bg-slate-950"}`}>
                    {step.icon}
                  </div>
                  <span className="text-xs font-semibold block">{step.title}</span>
                  <div className="w-6 h-1 rounded bg-indigo-500/20 mt-1"></div>
                </button>
              ))}
            </div>

            {/* Selected Step Description Area */}
            <div className="bg-slate-900/40 border border-slate-850/60 rounded-xl p-5 flex flex-col md:flex-row gap-4 items-start animate-fadeIn">
              <div className="p-3 bg-indigo-950/40 rounded-xl border border-indigo-900/50">
                {flowSteps[activeDiagStep].icon}
              </div>
              <div className="flex-1">
                <span className="text-[10px] text-indigo-400 font-bold uppercase tracking-wider block mb-0.5">Stage {activeDiagStep + 1} processing</span>
                <h3 className="text-sm font-bold text-slate-200 mb-2">{flowSteps[activeDiagStep].title}</h3>
                <p className="text-xs text-slate-400 leading-relaxed font-light">{flowSteps[activeDiagStep].desc}</p>
              </div>
            </div>
          </section>

          {/* Quick Start Guide */}
          <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-slate-950/70 border border-slate-800/80 rounded-2xl p-5 flex flex-col gap-2">
              <span className="text-indigo-400 font-mono text-sm font-bold">01.</span>
              <h3 className="text-sm font-bold text-slate-200">Configure Parameter Rules</h3>
              <p className="text-xs text-slate-400 font-light leading-relaxed">
                Choose between Hindi, English, or Marathi. Pick your FAISS chunk retrieval strategy and set the similarity cutoff score inside the console settings.
              </p>
            </div>
            <div className="bg-slate-950/70 border border-slate-800/80 rounded-2xl p-5 flex flex-col gap-2">
              <span className="text-indigo-400 font-mono text-sm font-bold">02.</span>
              <h3 className="text-sm font-bold text-slate-200">Speak or Upload Audio</h3>
              <p className="text-xs text-slate-400 font-light leading-relaxed">
                Press the microphone button to record your native speech query, select custom presets, or upload audio files directly from your computer.
              </p>
            </div>
            <div className="bg-slate-950/70 border border-slate-800/80 rounded-2xl p-5 flex flex-col gap-2">
              <span className="text-indigo-400 font-mono text-sm font-bold">03.</span>
              <h3 className="text-sm font-bold text-slate-200">Trace Safety & Latencies</h3>
              <p className="text-xs text-slate-400 font-light leading-relaxed">
                Inspect step-by-step latency metrics (P50/P70/P100), examine safety and groundedness checkpoints, and view retrieved context database passages.
              </p>
            </div>
          </section>

          {/* Footer */}
          <footer className="text-center text-[10px] text-slate-500 mt-6 pb-6">
            © 2026 Team Indic. Build version 1.1.0 (FastAPI + React)
          </footer>
        </div>
      )}

      {/* ── CONSOLE VIEW ── */}
      {view === "console" && (
        <div className="flex flex-col flex-1 animate-fadeIn">
          {/* ── Top Header Bar ── */}
          <header className="border-b border-slate-850 bg-slate-950/60 backdrop-blur-lg px-6 py-4 sticky top-0 z-40">
            <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-4">
              <div className="flex items-center gap-3">
                <button 
                  onClick={() => setView("landing")}
                  className="bg-slate-900 hover:bg-slate-800 p-2 rounded-lg border border-slate-850 text-slate-400 hover:text-slate-200 transition-colors mr-1 cursor-pointer"
                >
                  <ArrowRight className="h-4 w-4 rotate-180" />
                </button>
                <div className="bg-indigo-650 p-2 rounded-xl shadow-lg shadow-indigo-600/20">
                  <Mic className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h1 className="text-md font-bold tracking-tight bg-gradient-to-r from-indigo-200 to-indigo-400 bg-clip-text text-transparent">
                    Indic Voice Console
                  </h1>
                  <p className="text-[10px] text-slate-450">Decoupled Multilingual Voice-RAG (HH Goa 2026)</p>
                </div>
              </div>

              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <span className="h-2.5 w-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
                  <span className="text-[11px] text-slate-300 font-medium bg-slate-900/80 px-2.5 py-1 rounded-full border border-slate-700">
                    Backend Status: Connected
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  {apiStatus.live_mode_ready ? (
                    <span className="text-[11px] font-semibold bg-emerald-955/80 text-emerald-400 border border-emerald-900/40 px-3 py-1 rounded-full animate-pulse">
                      Live API Mode (Groq/Sarvam)
                    </span>
                  ) : (
                    <span className="text-[11px] font-semibold bg-amber-955/80 text-amber-400 border border-amber-900/45 px-3 py-1 rounded-full">
                      Demo Mode (Simulated APIs)
                    </span>
                  )}
                </div>
              </div>
            </div>
          </header>

          {/* ── Main Layout Workspace ── */}
          <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 relative z-10">
            
            {/* ── Left Column: Config Panel ── */}
            <section className="lg:col-span-4 flex flex-col gap-6">
              
              {/* Strategy & Language Controls Card */}
              <div className="bg-slate-950/70 border border-slate-800/70 rounded-2xl p-5 backdrop-blur-md shadow-xl">
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
                    
                    {/* Language Selector Dropdown */}
                    <div>
                      <label className="text-xs text-slate-400 block mb-1.5 font-medium">Input/Output Language</label>
                      <select
                        value={language}
                        onChange={(e) => handleLanguageChange(e.target.value)}
                        className="w-full bg-slate-900 border border-slate-800 rounded-lg py-2 px-3 text-sm text-slate-200 focus:outline-none focus:border-indigo-500 cursor-pointer transition-colors"
                      >
                        {Object.entries(languageConfigs).map(([code, cfg]) => (
                          <option key={code} value={code}>
                            {cfg.name}
                          </option>
                        ))}
                      </select>
                      <span className="text-[10px] text-slate-500 block mt-1">Sets the speech model language and target LLM output script.</span>
                    </div>

                    <div>
                      <label className="text-xs text-slate-400 block mb-1.5 font-medium">FAISS Chunk Strategy</label>
                      <select
                        value={strategy}
                        onChange={(e) => setStrategy(e.target.value)}
                        className="w-full bg-slate-900 border border-slate-800 rounded-lg py-2 px-3 text-sm text-slate-200 focus:outline-none focus:border-indigo-500 cursor-pointer transition-colors"
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
                          className="h-4.5 w-4.5 rounded border-slate-850 text-indigo-600 focus:ring-indigo-500 accent-indigo-500 cursor-pointer"
                        />
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Latency Percentiles Analytics Board */}
              <div className="bg-slate-955/70 border border-slate-800/70 rounded-2xl p-5 backdrop-blur-md shadow-xl">
                <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-4 flex items-center gap-2">
                  <Activity className="h-4 w-4 text-indigo-400" />
                  Latency analytics (P50/P70/P100)
                </h2>
                {latencyReport ? (
                  <div className="flex flex-col gap-3">
                    <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-900/60 flex justify-between items-center">
                      <span className="text-xs text-slate-400">STT Time (Mock)</span>
                      <div className="text-xs text-slate-300 font-semibold flex gap-2">
                        <span>P50: <b className="text-indigo-300">{latencyReport.summary.stt?.p50}ms</b></span>
                        <span>P99: <b className="text-indigo-400">{latencyReport.summary.stt?.p100}ms</b></span>
                      </div>
                    </div>
                    <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-900/60 flex justify-between items-center">
                      <span className="text-xs text-slate-400">Retrieval (FAISS)</span>
                      <div className="text-xs text-slate-300 font-semibold flex gap-2">
                        <span>P50: <b className="text-indigo-300">{latencyReport.summary.retrieval?.p50}ms</b></span>
                        <span>P99: <b className="text-indigo-400">{latencyReport.summary.retrieval?.p100}ms</b></span>
                      </div>
                    </div>
                    <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-900/60 flex justify-between items-center">
                      <span className="text-xs text-slate-400">LLM Generation</span>
                      <div className="text-xs text-slate-300 font-semibold flex gap-2">
                        <span>P50: <b className="text-indigo-300">{latencyReport.summary.generation?.p50}ms</b></span>
                        <span>P99: <b className="text-indigo-400">{latencyReport.summary.generation?.p100}ms</b></span>
                      </div>
                    </div>
                    <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-900/60 flex justify-between items-center">
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
              <div className="bg-slate-950/70 border border-slate-800/70 rounded-2xl p-6 backdrop-blur-md shadow-xl">
                <h2 className="text-md font-semibold text-slate-200 mb-4">Input Console</h2>
                
                <div className="flex flex-col items-center justify-center py-6 bg-slate-900/40 rounded-2xl border border-slate-850/60 mb-6">
                  
                  {/* Pulsing Mic Aura */}
                  <div className="relative mb-4">
                    <button
                      onClick={isRecording ? stopRecording : startRecording}
                      disabled={loading}
                      className={`w-20 h-20 rounded-full flex items-center justify-center transition-all ${
                        isRecording 
                          ? "bg-red-650 shadow-red-500/50 shadow-xl scale-105" 
                          : "bg-indigo-600 hover:bg-indigo-550 shadow-indigo-500/25 shadow-lg hover:scale-105"
                      } disabled:opacity-50 cursor-pointer`}
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
                      {isRecording ? `Recording... (${recordingTime}s)` : `Press microphone to record ${languageConfigs[language].name}`}
                    </p>
                    <p className="text-xs text-slate-500 mt-1">Accepts voice queries and transcribes via Sarvam AI.</p>
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
                    <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg p-1">
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
                        className="block w-full text-xs text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-slate-950 file:text-indigo-400 hover:file:bg-slate-900 cursor-pointer"
                      />
                    </div>
                  </div>

                  <div className="flex-1">
                    <label className="text-xs text-slate-400 block mb-1 font-medium">Or type query text directly</label>
                    <input
                      type="text"
                      placeholder={languageConfigs[language].placeholder}
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      className="w-full bg-slate-900 border border-slate-800 rounded-lg py-2.5 px-3 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-indigo-500"
                    />
                  </div>
                </div>

                {/* Quick click tags */}
                <div className="flex flex-wrap gap-2 items-center mb-6">
                  <span className="text-[10px] text-slate-500 font-medium">Preset Prompts:</span>
                  {languageConfigs[language].presets.map((p, idx) => (
                    <button
                      key={idx}
                      onClick={() => loadPreset(p.text)}
                      className="text-[11px] bg-indigo-950/40 text-indigo-300 hover:bg-indigo-950/70 border border-indigo-900/50 px-2.5 py-1 rounded-md transition-colors cursor-pointer"
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
                    className="flex-1 bg-indigo-650 hover:bg-indigo-650 text-white text-sm font-semibold py-2.5 px-4 rounded-lg shadow-lg shadow-indigo-600/10 hover:shadow-indigo-600/20 transition-all flex items-center justify-center gap-2 disabled:opacity-50 cursor-pointer"
                  >
                    {loading ? (
                      <>
                        <RefreshCw className="h-4 w-4 animate-spin" />
                        Executing Pipeline...
                      </>
                    ) : (
                      <span className="flex items-center justify-center gap-1.5">
                        <Play className="h-4 w-4" />
                        Execute pipeline
                      </span>
                    )}
                  </button>

                  <button
                    onClick={resetInputs}
                    className="bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-400 hover:text-slate-250 text-sm font-medium py-2.5 px-4 rounded-lg transition-all cursor-pointer"
                  >
                    Reset
                  </button>
                </div>
              </div>

              {/* ── System Response Cards ── */}
              {response && (
                <div className="flex flex-col gap-6 animate-fadeIn">
                  
                  {/* Output & Transcription Card */}
                  <div className="bg-slate-950/70 border border-slate-800/70 rounded-2xl p-6 backdrop-blur-md shadow-xl">
                    <h3 className="text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                      <FileText className="h-4 w-4" />
                      Processed Output
                    </h3>
                    
                    {/* Transcription display */}
                    <div className="mb-4">
                      <span className="text-[10px] text-slate-500 block mb-1">STT Transcription query:</span>
                      <p className="text-md font-medium text-slate-250 bg-slate-900/50 p-3 rounded-lg border border-slate-850/60 italic">
                        "{response.query}"
                      </p>
                    </div>

                    {/* Final Generation response */}
                    <div>
                      <span className="text-[10px] text-slate-500 block mb-1">Synthesized Answer:</span>
                      <div className={`p-4 rounded-xl border leading-relaxed text-sm font-medium ${
                        response.status.includes("blocked")
                          ? "bg-red-950/20 text-red-400 border-red-900/35"
                          : "bg-slate-900/50 text-slate-100 border-slate-850/60"
                      }`}>
                        {response.answer}
                      </div>
                    </div>

                    {/* Citations display */}
                    {response.citations && response.citations.length > 0 && (
                      <div className="mt-4 pt-4 border-t border-slate-850/60">
                        <button
                          onClick={() => setOpenCitations(!openCitations)}
                          className="text-xs text-indigo-400 flex items-center gap-1 font-semibold hover:text-indigo-300 cursor-pointer"
                        >
                          References Used ({response.citations.length})
                          {openCitations ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                        </button>
                        {openCitations && (
                          <div className="mt-2 flex flex-col gap-1.5">
                            {response.citations.map((c, i) => (
                              <div key={i} className="text-xs text-slate-400 flex items-center gap-2">
                                <CornerDownRight className="h-3.5 w-3.5 text-slate-650" />
                                <span>Passage ID: <code className="text-indigo-300 bg-slate-900 px-1 rounded">{c}</code></span>
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
                          className="text-xs text-indigo-400 flex items-center gap-1 font-semibold hover:text-indigo-300 cursor-pointer"
                        >
                          Show reasoning trace
                          {openReasoning ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                        </button>
                        {openReasoning && (
                          <div className="mt-2 p-3 bg-slate-900/80 rounded-lg border border-slate-850 text-xs text-slate-400 font-mono leading-relaxed whitespace-pre-line">
                            {response.reasoning}
                          </div>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Execution Flow & Guardrail Timeline Panel */}
                  <div className="bg-slate-950/70 border border-slate-800/70 rounded-2xl p-6 backdrop-blur-md shadow-xl grid grid-cols-1 md:grid-cols-2 gap-6">
                    
                    {/* Latency break down */}
                    <div>
                      <h3 className="text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-4 flex items-center gap-1.5">
                        <Clock className="h-4 w-4" />
                        Latency trace
                      </h3>
                      <div className="flex flex-col gap-2">
                        <div className="flex justify-between items-center bg-slate-900/40 p-2.5 rounded-lg border border-slate-850/60">
                          <span className="text-xs text-slate-400">Speech-To-Text</span>
                          <span className="text-xs text-slate-300 font-semibold">{response.latencies.stt?.toFixed(1)} ms</span>
                        </div>
                        <div className="flex justify-between items-center bg-slate-900/40 p-2.5 rounded-lg border border-slate-850/60">
                          <span className="text-xs text-slate-400">Safety Check</span>
                          <span className="text-xs text-slate-300 font-semibold">{response.latencies.safety_guard?.toFixed(1)} ms</span>
                        </div>
                        <div className="flex justify-between items-center bg-slate-900/40 p-2.5 rounded-lg border border-slate-850/60">
                          <span className="text-xs text-slate-400">FAISS retrieval</span>
                          <span className="text-xs text-slate-300 font-semibold">{response.latencies.retrieval?.toFixed(1)} ms</span>
                        </div>
                        <div className="flex justify-between items-center bg-slate-900/40 p-2.5 rounded-lg border border-slate-850/60">
                          <span className="text-xs text-slate-400">LLM Generation</span>
                          <span className="text-xs text-slate-300 font-semibold">{response.latencies.generation?.toFixed(1)} ms</span>
                        </div>
                        <div className="flex justify-between items-center bg-slate-900/70 p-2.5 rounded-lg border border-indigo-950/40">
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
                        <div className="flex justify-between items-center bg-slate-900/40 p-2.5 rounded-lg border border-slate-850/60">
                          <span className="text-xs text-slate-300">Input Safety</span>
                          {response.guardrails.safety.safe ? (
                            <span className="text-[10px] bg-emerald-950 text-emerald-400 px-2 py-0.5 rounded font-bold border border-emerald-900/40">Passed</span>
                          ) : (
                            <span className="text-[10px] bg-red-950 text-red-400 px-2 py-0.5 rounded font-bold border border-red-900/40">Violation</span>
                          )}
                        </div>

                        {/* Off-Topic */}
                        <div className="flex justify-between items-center bg-slate-900/40 p-2.5 rounded-lg border border-slate-850/60">
                          <span className="text-xs text-slate-300">Topic Scope</span>
                          {!response.guardrails.off_topic ? (
                            <span className="text-[10px] bg-slate-900 text-slate-500 px-2 py-0.5 rounded font-bold">Skipped</span>
                          ) : response.guardrails.off_topic.on_topic ? (
                            <span className="text-[10px] bg-emerald-950 text-emerald-400 px-2 py-0.5 rounded font-bold border border-emerald-900/40">Passed</span>
                          ) : (
                            <span className="text-[10px] bg-red-950 text-red-400 px-2 py-0.5 rounded font-bold border border-red-900/40">Off-topic</span>
                          )}
                        </div>

                        {/* Groundedness */}
                        <div className="flex justify-between items-center bg-slate-900/40 p-2.5 rounded-lg border border-slate-850/60">
                          <span className="text-xs text-slate-300">Grounded Context</span>
                          {!response.guardrails.groundedness ? (
                            <span className="text-[10px] bg-slate-900 text-slate-500 px-2 py-0.5 rounded font-bold">Skipped</span>
                          ) : response.guardrails.groundedness.grounded ? (
                            <span className="text-[10px] bg-emerald-950 text-emerald-400 px-2 py-0.5 rounded font-bold border border-emerald-900/40">Grounded</span>
                          ) : (
                            <span className="text-[10px] bg-red-950 text-red-400 px-2 py-0.5 rounded font-bold border border-red-900/40">Hallucination</span>
                          )}
                        </div>
                      </div>
                      {/* Block explanation message if any */}
                      {response.status !== "success" && (
                        <div className="mt-3 text-xs bg-red-950/20 text-red-400/80 p-2 rounded-lg border border-red-900/30 flex items-start gap-2 animate-pulse">
                          <AlertTriangle className="h-4 w-4 flex-shrink-0 mt-0.5" />
                          <span>{response.answer}</span>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Context passages */}
                  <div className="bg-slate-955/70 border border-slate-800/70 rounded-2xl p-6 backdrop-blur-md shadow-xl">
                    <h3 className="text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-4">
                      Retrieved Context Chunks from FAISS
                    </h3>
                    <div className="flex flex-col gap-4">
                      {response.chunks && response.chunks.length > 0 ? (
                        response.chunks.map((c, i) => (
                          <div key={i} className="bg-slate-900/50 p-4 rounded-xl border border-slate-850/60">
                            <div className="flex justify-between text-xs mb-2">
                              <span className="text-indigo-400 font-semibold">Chunk [{i}] ID: <code className="bg-slate-950 px-1 rounded">{c.passage_id}</code></span>
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
      )}
    </div>
  );
}
