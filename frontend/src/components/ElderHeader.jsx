import { Shield, Trees, Beef, NotebookPen } from "lucide-react";

export default function ElderHeader({ activeTab, onTabChange }) {
  return (
    <header className="elder-header">
      <div className="brand-block">
        <div className="brand-title">
          <Shield size={42} strokeWidth={2.4} aria-hidden="true" />
          <h1>Haran AI</h1>
        </div>
        <p className="brand-subtitle">Smart guardian for the grove and the barn</p>
        <p className="brand-english">Autonomous Agro-Pastoral Guardian</p>
      </div>

      <nav className="elder-tabs" aria-label="Haran sections">
        <button
          type="button"
          className={`elder-tab field-tab ${activeTab === "field" ? "is-active" : ""}`}
          onClick={() => onTabChange("field")}
        >
          <Trees size={36} strokeWidth={2.3} aria-hidden="true" />
          <span>Grove watch (olive check)</span>
        </button>
        <button
          type="button"
          className={`elder-tab barn-tab ${activeTab === "barn" ? "is-active" : ""}`}
          onClick={() => onTabChange("barn")}
        >
          <Beef size={36} strokeWidth={2.3} aria-hidden="true" />
          <span>Barn watch (cow camera)</span>
        </button>
        <button
          type="button"
          className={`elder-tab book-tab ${activeTab === "book" ? "is-active" : ""}`}
          onClick={() => onTabChange("book")}
        >
          <NotebookPen size={36} strokeWidth={2.3} aria-hidden="true" />
          <span>Farm book (checks)</span>
        </button>
      </nav>
    </header>
  );
}
