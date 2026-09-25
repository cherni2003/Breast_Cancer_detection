import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload, FlaskConical, AlertCircle, CheckCircle,
  ImageIcon, X, Download, Activity, Cpu, Clock,
  Crosshair, LayoutGrid, Info, Heart
} from "lucide-react";

const API_BASE = "http://localhost:8000";

const GlobalStyles = () => (
  <style>{`
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg1:    #fdf4f9;
      --bg2:    #fff8fc;
      --bg3:    #fce7f3;
      --bg4:    #f8d7ec;
      --border: rgba(236,72,153,0.12);
      --border2:rgba(168,85,247,0.2);
      --text:   #2d1b36;
      --text2:  #7c5a8a;
      --text3:  #b48ec4;
      --accent: #ec4899;
      --accent2:#a855f7;
      --accentg: linear-gradient(135deg,#ec4899,#a855f7);
      --malin:  #e11d48;
      --malin2: #fb7185;
      --benin:  #059669;
      --benin2: #34d399;
      --warn:   #d97706;
      --radius1:10px;
      --radius2:18px;
      --radius3:28px;
      --shadow: 0 4px 24px rgba(236,72,153,0.10), 0 1px 4px rgba(168,85,247,0.07);
    }

    html,body,#root {
      min-height:100vh;
      background: var(--bg1);
      color: var(--text);
      font-family:'DM Sans',sans-serif;
      font-size:14px;
      line-height:1.6;
      -webkit-font-smoothing:antialiased;
    }

    /* ── background petals ── */
    body::before {
      content:'';
      position:fixed; inset:0; pointer-events:none; z-index:0;
      background:
        radial-gradient(ellipse 600px 400px at 10% 20%, rgba(236,72,153,0.06) 0%, transparent 70%),
        radial-gradient(ellipse 500px 500px at 90% 80%, rgba(168,85,247,0.07) 0%, transparent 70%),
        radial-gradient(ellipse 300px 300px at 50% 50%, rgba(244,114,182,0.04) 0%, transparent 70%);
    }

    .layout { display:flex; min-height:100vh; position:relative; z-index:1; }

    /* ── Sidebar ── */
    .sidebar {
      width:230px; flex-shrink:0;
      background: rgba(255,255,255,0.75);
      backdrop-filter: blur(20px);
      border-right:1px solid var(--border);
      display:flex; flex-direction:column;
      padding:0;
      position:fixed; top:0; left:0; bottom:0;
      box-shadow: 2px 0 20px rgba(236,72,153,0.06);
    }

    .sidebar-logo {
      padding:28px 24px 24px;
      border-bottom:1px solid var(--border);
    }
    .logo-brand {
      display:flex; align-items:center; gap:10px; margin-bottom:4px;
    }
    .logo-ribbon {
      width:8px; height:28px; border-radius:4px;
      background: var(--accentg);
      flex-shrink:0;
    }
    .logo-text {
      font-family:'Playfair Display',serif;
      font-weight:900; font-size:18px;
      background: var(--accentg);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
      letter-spacing:-0.5px;
    }
    .logo-sub {
      font-size:10px; color:var(--text3);
      letter-spacing:2px; text-transform:uppercase;
      padding-left:18px;
    }

    .nav-section {
      padding:16px 12px 8px;
      font-size:9px; font-weight:600; letter-spacing:1.5px;
      text-transform:uppercase; color:var(--text3);
    }

    .nav-item {
      display:flex; align-items:center; gap:10px;
      padding:10px 16px; margin:1px 8px;
      font-size:13px; font-weight:500;
      color:var(--text2); cursor:pointer;
      transition:all 0.2s;
      border-radius:var(--radius1);
      border-left:2px solid transparent;
    }
    .nav-item.active {
      color:var(--accent);
      background: rgba(236,72,153,0.08);
      border-left-color:var(--accent);
    }
    .nav-item:hover:not(.active) {
      color:var(--text);
      background: rgba(236,72,153,0.05);
    }

    .sidebar-footer {
      margin-top:auto; padding:20px;
      border-top:1px solid var(--border);
    }
    .auc-badge {
      font-family:'Space Mono',monospace; font-size:11px;
      color:var(--accent2);
      background: rgba(168,85,247,0.08);
      border:1px solid rgba(168,85,247,0.2);
      border-radius:var(--radius1);
      padding:8px 12px; text-align:center;
      line-height:1.7;
    }

    /* ── Main ── */
    .main { margin-left:230px; flex:1; padding:36px 40px; min-height:100vh; }

    .page-header { margin-bottom:32px; }
    .page-title {
      font-family:'Playfair Display',serif;
      font-size:26px; font-weight:900;
      background: var(--accentg);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
      margin-bottom:4px;
    }
    .page-sub { font-size:12px; color:var(--text3); }

    /* ── Cards ── */
    .card {
      background: rgba(255,255,255,0.80);
      backdrop-filter: blur(12px);
      border:1px solid var(--border);
      border-radius:var(--radius2);
      box-shadow: var(--shadow);
    }
    .card-title {
      font-size:10px; font-weight:600;
      text-transform:uppercase; letter-spacing:1.2px;
      color:var(--text3); margin-bottom:14px;
    }

    /* ── Dropzone ── */
    .dropzone {
      padding:40px 24px; text-align:center; cursor:pointer;
      transition:all 0.2s;
      background: var(--bg3);
      border-radius:var(--radius2) var(--radius2) 0 0;
    }
    .dropzone:hover,.dropzone.active {
      background:var(--bg4);
    }
    .dropzone-icon {
      width:52px; height:52px; border-radius:14px;
      background: rgba(236,72,153,0.10);
      border:1px solid rgba(236,72,153,0.20);
      display:flex; align-items:center; justify-content:center;
      margin:0 auto 16px; color:var(--accent);
    }

    /* ── Forms ── */
    .form-group { margin-bottom:14px; }
    .form-group label {
      display:block; font-size:10px; font-weight:600;
      text-transform:uppercase; letter-spacing:0.8px;
      color:var(--text3); margin-bottom:6px;
    }
    .form-group textarea {
      width:100%;
      background:var(--bg3);
      border:1px solid var(--border2);
      border-radius:var(--radius1);
      color:var(--text); padding:10px 14px;
      font-family:'DM Sans',sans-serif; font-size:13px;
      outline:none; transition:border-color 0.2s;
    }
    .form-group textarea:focus { border-color:var(--accent); }
    .form-group textarea::placeholder { color:var(--text3); }

    /* ── Buttons ── */
    .btn {
      display:inline-flex; align-items:center; gap:8px;
      border-radius:50px; padding:11px 20px;
      font-family:'DM Sans',sans-serif; font-size:13px; font-weight:600;
      cursor:pointer; transition:all 0.2s; border:none;
    }
    .btn-primary {
      background: var(--accentg); color:#fff;
      box-shadow: 0 4px 20px rgba(236,72,153,0.30);
    }
    .btn-primary:hover:not(:disabled) {
      transform:translateY(-1px);
      box-shadow:0 6px 28px rgba(236,72,153,0.40);
    }
    .btn-primary:active:not(:disabled) { transform:scale(0.98); }
    .btn-primary:disabled { opacity:0.45; cursor:not-allowed; }

    .btn-ghost {
      background:transparent;
      border:1.5px solid rgba(236,72,153,0.25);
      color:var(--accent);
      border-radius:50px;
    }
    .btn-ghost:hover { background:rgba(236,72,153,0.06); }

    /* ── Spinner ── */
    .spinner {
      width:14px; height:14px;
      border:2px solid rgba(255,255,255,0.3);
      border-top-color:#fff;
      border-radius:50%;
      animation:spin 0.7s linear infinite; flex-shrink:0;
    }
    @keyframes spin { to { transform:rotate(360deg); } }

    /* ── Result ── */
    .result-panel {
      background: rgba(255,255,255,0.85);
      backdrop-filter:blur(12px);
      border:1px solid var(--border);
      border-radius:var(--radius2);
      box-shadow:var(--shadow);
      padding:24px;
    }
    .result-verdict {
      font-family:'Playfair Display',serif;
      font-size:24px; font-weight:900;
    }
    .result-verdict.malin  { color:var(--malin2); }
    .result-verdict.benin  { color:var(--benin2); }

    /* ── Prob bar ── */
    .prob-bar {
      background:var(--bg3); border-radius:99px; overflow:hidden; height:9px;
    }
    .prob-fill {
      height:100%; border-radius:99px;
      transition:width 0.9s cubic-bezier(0.22,1,0.36,1);
    }

    /* ── Meta cell ── */
    .meta-cell {
      background:var(--bg3);
      border:1px solid var(--border);
      border-radius:var(--radius1); padding:10px 14px;
    }
    .meta-label {
      font-size:10px; color:var(--text3);
      text-transform:uppercase; letter-spacing:0.8px;
      margin-bottom:4px; display:flex; align-items:center; gap:4px;
    }
    .meta-value {
      font-family:'Space Mono',monospace; font-size:12px; color:var(--text);
      overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
    }

    /* ── Empty ── */
    .empty-state { text-align:center; padding:56px 24px; color:var(--text3); }

    /* ── Heatmap ── */
    .heatmap-wrap { background:var(--bg3); border-radius:var(--radius1); overflow:hidden; }
    .heatmap-wrap img { width:100%; display:block; object-fit:contain; }

    /* ── Floating petals decoration ── */
    @keyframes floatA { 0%,100%{transform:translateY(0) rotate(-15deg)} 50%{transform:translateY(-12px) rotate(-15deg)} }
    @keyframes floatB { 0%,100%{transform:translateY(0) rotate(30deg)}  50%{transform:translateY(-18px) rotate(30deg)} }
    @keyframes floatC { 0%,100%{transform:translateY(0) rotate(5deg)}   50%{transform:translateY(-8px)  rotate(5deg)}  }
    @keyframes pulse  { 0%,100%{transform:scale(1)} 50%{transform:scale(1.2)} }

    .deco { position:fixed; pointer-events:none; z-index:0; }
  `}</style>
);

// ── SVG Decorations ───────────────────────────────────────────────────────────
const Deco = () => (
  <>
    {/* top-right flower */}
    <svg className="deco" style={{top:30,right:40,opacity:0.18,animation:"floatB 7s ease-in-out infinite"}} width="54" height="54" viewBox="0 0 54 54">
      <ellipse cx="27" cy="27" rx="10" ry="20" fill="#ec4899" transform="rotate(0 27 27)"/>
      <ellipse cx="27" cy="27" rx="10" ry="20" fill="#a855f7" transform="rotate(60 27 27)"/>
      <ellipse cx="27" cy="27" rx="10" ry="20" fill="#f472b6" transform="rotate(120 27 27)"/>
      <circle  cx="27" cy="27" r="6" fill="#fde68a"/>
    </svg>
    {/* bottom-left flower */}
    <svg className="deco" style={{bottom:60,left:50,opacity:0.14,animation:"floatA 9s ease-in-out infinite"}} width="44" height="44" viewBox="0 0 44 44">
      <ellipse cx="22" cy="22" rx="8" ry="18" fill="#f472b6" transform="rotate(0 22 22)"/>
      <ellipse cx="22" cy="22" rx="8" ry="18" fill="#c084fc" transform="rotate(72 22 22)"/>
      <ellipse cx="22" cy="22" rx="8" ry="18" fill="#ec4899" transform="rotate(144 22 22)"/>
      <circle  cx="22" cy="22" r="5" fill="#fef08a"/>
    </svg>
    {/* top-left ribbon */}
    <svg className="deco" style={{top:120,left:260,opacity:0.12,animation:"floatC 6s ease-in-out infinite"}} width="28" height="36" viewBox="0 0 40 50">
      <path fill="#a855f7" d="M20 12 C10 12 10 0 20 0 C30 0 30 12 20 12Z M20 12 C8 20 2 32 20 32 C38 32 32 20 20 12Z"/>
      <path fill="#a855f7" d="M20 32 L15 50 L20 45 L25 50 Z"/>
    </svg>
    {/* hearts */}
    <svg className="deco" style={{top:"35%",right:30,opacity:0.15,animation:"pulse 4s ease-in-out infinite"}} width="22" height="22" viewBox="0 0 24 24">
      <path fill="#ec4899" d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
    </svg>
    <svg className="deco" style={{bottom:"25%",right:60,opacity:0.12,animation:"pulse 5.5s ease-in-out infinite"}} width="18" height="18" viewBox="0 0 24 24">
      <path fill="#a855f7" d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
    </svg>
  </>
);

// ── Sidebar ───────────────────────────────────────────────────────────────────
const Sidebar = ({ active }) => (
  <aside className="sidebar">
    <div className="sidebar-logo">
      <div className="logo-brand">
        <div className="logo-ribbon"/>
        <span className="logo-text">Mammo·CLIP</span>
      </div>
      <div className="logo-sub">Diagnostic IA</div>
    </div>
    <div className="nav-section">Navigation</div>
    {[
      { icon:<LayoutGrid size={14}/>,   label:"Tableau de bord", id:"dashboard" },
      { icon:<FlaskConical size={14}/>, label:"Analyser",        id:"analyze"   },
      { icon:<Activity size={14}/>,     label:"Historique",      id:"history"   },
      { icon:<Info size={14}/>,         label:"À propos",        id:"about"     },
    ].map(n => (
      <div key={n.id} className={`nav-item ${active===n.id?"active":""}`}>
        {n.icon}{n.label}
      </div>
    ))}
    <div className="sidebar-footer">
      <div className="auc-badge">
        AUC 0.796<br/>
        <span style={{color:"var(--text3)",fontSize:9}}>EfficientNet-B5 · GradCAM++</span>
      </div>
      <div style={{marginTop:12,textAlign:"center",fontSize:11,color:"var(--text3)"}}>
        🎗️ Détection précoce sauve des vies
      </div>
    </div>
  </aside>
);

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const [file,    setFile]    = useState(null);
  const [preview, setPreview] = useState(null);
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);
  const [notes,   setNotes]   = useState("");

  const onDrop = useCallback((accepted) => {
    const f = accepted[0];
    if (!f) return;
    setFile(f); setResult(null); setError(null);
    const reader = new FileReader();
    reader.onload = e => setPreview(e.target.result);
    reader.readAsDataURL(f);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept:{ "image/*":[".png",".jpg",".jpeg",".tiff",".bmp",".dcm"] },
    maxFiles:1,
  });

  const clearAll = () => { setFile(null); setPreview(null); setResult(null); setError(null); setNotes(""); };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${API_BASE}/analyze`, { method:"POST", body:fd });
      if (!res.ok) { const e=await res.json().catch(()=>({})); throw new Error(e.detail||`HTTP ${res.status}`); }
      setResult(await res.json());
    } catch(e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const downloadImage = async (url, filename) => {
    try {
      const blob = await (await fetch(url)).blob();
      const a = Object.assign(document.createElement("a"),{
        href:URL.createObjectURL(blob), download:filename||"gradcam_result.png"
      });
      document.body.appendChild(a); a.click();
      document.body.removeChild(a); URL.revokeObjectURL(a.href);
    } catch(e) { alert("Erreur téléchargement : "+e.message); }
  };

  const isMasse = result?.prediction === "MASSE";
  const prob    = result ? result.prob*100 : 0;

  return (
    <>
      <GlobalStyles/>
      <Deco/>
      <div className="layout">
        <Sidebar active="analyze"/>

        <main className="main">
          <div className="page-header">
            <div className="page-title">Analyser une mammographie</div>
            <div className="page-sub">Classification par IA · GradCAM++ multi-échelle · 4 Février — Journée mondiale contre le cancer 🎗️</div>
          </div>

          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:22,alignItems:"start"}}>

            {/* ── Left ── */}
            <div style={{display:"flex",flexDirection:"column",gap:16}}>

              <div className="card" style={{padding:0,overflow:"hidden"}}>
                <div {...getRootProps()} className={`dropzone ${isDragActive?"active":""}`}
                  style={{borderBottom:"1px solid var(--border)"}}>
                  <input {...getInputProps()}/>
                  {preview ? (
                    <div style={{position:"relative",display:"inline-block"}}>
                      <img src={preview} alt="preview"
                        style={{maxHeight:200,maxWidth:"100%",borderRadius:10,objectFit:"contain"}}/>
                      <button onClick={e=>{e.stopPropagation();clearAll();}} style={{
                        position:"absolute",top:-8,right:-8,
                        background:"white", border:"1.5px solid rgba(236,72,153,0.3)",
                        borderRadius:"50%", width:26,height:26, cursor:"pointer",
                        display:"flex",alignItems:"center",justifyContent:"center",
                        color:"var(--accent)", boxShadow:"0 2px 8px rgba(236,72,153,0.15)"
                      }}><X size={12}/></button>
                    </div>
                  ) : (
                    <>
                      <div className="dropzone-icon"><Upload size={22}/></div>
                      <p style={{color:"var(--text)",fontWeight:600,marginBottom:6,fontSize:15}}>
                        {isDragActive ? "Déposez l'image ici ✨" : "Glissez ou cliquez pour charger"}
                      </p>
                      <p style={{color:"var(--text3)",fontSize:12}}>PNG · JPG · TIFF · DICOM</p>
                    </>
                  )}
                </div>
                {file && (
                  <div style={{padding:"10px 18px",display:"flex",alignItems:"center",gap:8,background:"rgba(252,231,243,0.4)"}}>
                    <ImageIcon size={13} style={{color:"var(--accent)",flexShrink:0}}/>
                    <span style={{fontSize:12,color:"var(--text2)",flex:1,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{file.name}</span>
                    <span style={{fontSize:11,color:"var(--text3)",flexShrink:0,fontFamily:"'Space Mono',monospace"}}>{(file.size/1024).toFixed(0)} Ko</span>
                  </div>
                )}
              </div>

              <div className="card" style={{padding:20}}>
                <div className="card-title">Notes cliniques</div>
                <div className="form-group" style={{marginBottom:0}}>
                  <textarea rows={3} placeholder="Observations, contexte clinique…"
                    value={notes} onChange={e=>setNotes(e.target.value)} style={{resize:"vertical"}}/>
                </div>
              </div>

              <div style={{display:"flex",gap:10}}>
                <button className="btn btn-primary"
                  style={{flex:1,justifyContent:"center",padding:"13px"}}
                  onClick={handleAnalyze} disabled={!file||loading}>
                  {loading
                    ? <><div className="spinner"/>Analyse en cours…</>
                    : <><FlaskConical size={15}/>Lancer l'analyse</>}
                </button>
                {file && (
                  <button className="btn btn-ghost" onClick={clearAll} style={{padding:"13px 16px"}}>
                    <X size={14}/>
                  </button>
                )}
              </div>

              {/* How it works */}
              <div className="card" style={{padding:20}}>
                <div className="card-title">Comment ça fonctionne</div>
                {[
                  [<Upload size={14}/>,       "Import",    "Chargez votre mammographie (PNG, DICOM…)"],
                  [<FlaskConical size={14}/>, "Analyse",   "EfficientNet-B5 détecte les masses suspectes"],
                  [<Activity size={14}/>,     "GradCAM++", "Visualisation multi-échelle des zones d'intérêt"],
                  [<Crosshair size={14}/>,    "Résultat",  "Score de probabilité + bounding box localisée"],
                ].map(([ico,title,desc]) => (
                  <div key={title} style={{display:"flex",gap:12,marginBottom:14}}>
                    <div style={{
                      width:32,height:32,borderRadius:8,flexShrink:0,
                      background:"rgba(236,72,153,0.09)",
                      border:"1px solid rgba(236,72,153,0.15)",
                      display:"flex",alignItems:"center",justifyContent:"center",color:"var(--accent)"
                    }}>{ico}</div>
                    <div>
                      <p style={{fontWeight:600,color:"var(--text)",fontSize:13}}>{title}</p>
                      <p style={{color:"var(--text3)",fontSize:12,lineHeight:1.5}}>{desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* ── Right ── */}
            <AnimatePresence mode="wait">

              {error && (
                <motion.div key="error" className="card" style={{padding:22}}
                  initial={{opacity:0,x:14}} animate={{opacity:1,x:0}} exit={{opacity:0}}>
                  <div style={{display:"flex",gap:12,alignItems:"flex-start"}}>
                    <div style={{width:36,height:36,borderRadius:8,background:"rgba(251,113,133,0.1)",border:"1px solid rgba(251,113,133,0.2)",display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0}}>
                      <AlertCircle size={18} style={{color:"var(--malin2)"}}/>
                    </div>
                    <div>
                      <div style={{fontWeight:700,color:"var(--malin2)",marginBottom:4,fontSize:14}}>Erreur d'analyse</div>
                      <div style={{fontSize:13,color:"var(--text2)",lineHeight:1.6}}>{error}</div>
                    </div>
                  </div>
                </motion.div>
              )}

              {result && !error && (
                <motion.div key="result"
                  initial={{opacity:0,x:14}} animate={{opacity:1,x:0}} exit={{opacity:0}}
                  style={{display:"flex",flexDirection:"column",gap:16}}>

                  <div className="result-panel" style={{
                    background: isMasse
                      ? "linear-gradient(135deg,rgba(255,241,242,0.95),rgba(252,231,243,0.95))"
                      : "linear-gradient(135deg,rgba(240,253,250,0.95),rgba(243,232,255,0.95))"
                  }}>
                    {/* Verdict header */}
                    <div style={{display:"flex",alignItems:"center",gap:14,marginBottom:22}}>
                      <div style={{
                        width:48,height:48,borderRadius:12,flexShrink:0,
                        background: isMasse ? "rgba(251,113,133,0.12)" : "rgba(52,211,153,0.10)",
                        border:`1px solid ${isMasse?"rgba(251,113,133,0.25)":"rgba(52,211,153,0.22)"}`,
                        display:"flex",alignItems:"center",justifyContent:"center"
                      }}>
                        {isMasse
                          ? <AlertCircle size={22} color="var(--malin2)"/>
                          : <CheckCircle size={22} color="var(--benin2)"/>}
                      </div>
                      <div>
                        <div style={{fontSize:11,color:"var(--text3)",fontWeight:600,letterSpacing:2,textTransform:"uppercase",marginBottom:2}}>Résultat</div>
                        <div className={`result-verdict ${isMasse?"malin":"benin"}`}>{result.prediction}</div>
                        <div style={{fontSize:12,color:"var(--text3)",marginTop:2}}>Confiance : {result.confidence}</div>
                      </div>
                    </div>

                    {/* Prob bar */}
                    <div style={{marginBottom:20}}>
                      <div style={{display:"flex",justifyContent:"space-between",marginBottom:8}}>
                        <span style={{fontSize:12,color:"var(--text2)",fontWeight:500}}>Probabilité de masse</span>
                        <span style={{fontFamily:"'Space Mono',monospace",fontSize:16,fontWeight:700,color:isMasse?"var(--malin2)":"var(--benin2)"}}>
                          {prob.toFixed(1)}%
                        </span>
                      </div>
                      <div className="prob-bar">
                        <div className="prob-fill" style={{
                          width:`${prob}%`,
                          background:isMasse
                            ? "linear-gradient(to right,#fb7185,#e11d48)"
                            : "linear-gradient(to right,#34d399,#059669)"
                        }}/>
                      </div>
                      <div style={{position:"relative",height:16,marginTop:2}}>
                        <div style={{position:"absolute",left:"50%",top:0,width:1,height:8,background:"rgba(0,0,0,0.12)"}}/>
                        <div style={{position:"absolute",left:"50%",top:9,fontSize:9,color:"var(--text3)",transform:"translateX(-50%)"}}>50%</div>
                      </div>
                    </div>

                    {/* Meta */}
                    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10}}>
                      {[
                        [<Clock size={10}/>,     "Durée",   `${result.elapsed_s}s`],
                        [<Cpu size={10}/>,       "Mode",    result.demo_mode?"Démo":"Réel"],
                        [<Crosshair size={10}/>, "BBox",    result.bbox?`[${result.bbox.map(v=>Math.round(v)).join(",")}]`:"—"],
                        [<Activity size={10}/>,  "Modèle",  "B5-AUC0.796"],
                      ].map(([icon,label,val])=>(
                        <div className="meta-cell" key={label}>
                          <div className="meta-label"><span style={{color:"var(--accent)"}}>{icon}</span>{label}</div>
                          <div className="meta-value">{val}</div>
                        </div>
                      ))}
                    </div>

                    {isMasse && (
                      <div style={{
                        marginTop:16,padding:"10px 14px",
                        background:"rgba(217,119,6,0.07)",
                        border:"1px solid rgba(217,119,6,0.2)",
                        borderRadius:"var(--radius1)",fontSize:12,color:"var(--warn)",lineHeight:1.6
                      }}>
                        ⚕ Résultat indicatif uniquement. Consultez un radiologue pour toute décision médicale.
                      </div>
                    )}
                  </div>

                  {/* Heatmap */}
                  {result.heatmap_url && (
                    <div className="card" style={{padding:20}}>
                      <div className="card-title">Carte d'activation GradCAM++</div>
                      <div className="heatmap-wrap">
                        <img src={`${API_BASE}${result.heatmap_url}`} alt="GradCAM++ heatmap"
                          onError={e=>e.target.style.display="none"}/>
                      </div>
                      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginTop:12}}>
                        <span style={{fontSize:11,color:"var(--text3)"}}>Rouge = haute activation · Bleu = faible</span>
                        <button className="btn btn-ghost" style={{padding:"8px 16px",fontSize:12}}
                          onClick={()=>downloadImage(`${API_BASE}${result.heatmap_url}`,`gradcam_${result.job_id}.png`)}>
                          <Download size={13}/> Télécharger
                        </button>
                      </div>
                    </div>
                  )}
                </motion.div>
              )}

              {!result && !error && !loading && (
                <motion.div key="empty" className="card" initial={{opacity:0}} animate={{opacity:1}}>
                  <div className="empty-state">
                    <div style={{
                      width:64,height:64,borderRadius:20,
                      background:"rgba(236,72,153,0.08)",
                      border:"1px solid rgba(236,72,153,0.15)",
                      display:"flex",alignItems:"center",justifyContent:"center",
                      margin:"0 auto 16px", color:"var(--accent)"
                    }}>
                      <FlaskConical size={28}/>
                    </div>
                    <p style={{fontSize:14,color:"var(--text2)",fontWeight:500}}>Chargez une image pour démarrer</p>
                    <p style={{fontSize:12,color:"var(--text3)",marginTop:6}}>Mammographies PNG · JPG · DICOM supportées</p>
                  </div>
                </motion.div>
              )}

            </AnimatePresence>
          </div>
        </main>
      </div>
    </>
  );
}