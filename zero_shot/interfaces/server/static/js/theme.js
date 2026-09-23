// Theme switching: grey (default) <-> light, persisted in localStorage.

const STORAGE_KEY = "theme";

const SUN = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
const MOON = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';

function apply(theme) {
  if (theme === "light") {
    document.documentElement.setAttribute("data-theme", "light");
  } else {
    document.documentElement.removeAttribute("data-theme");
  }
  const button = document.getElementById("themeToggle");
  if (!button) return;
  // Show the theme you'd switch to.
  button.innerHTML = theme === "light" ? MOON : SUN;
  button.title = theme === "light" ? "Switch to dark" : "Switch to light";
  button.setAttribute("aria-label", button.title);
}

export function initTheme() {
  apply(localStorage.getItem(STORAGE_KEY) || "dark");
  const button = document.getElementById("themeToggle");
  if (!button) return;
  button.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
    const next = current === "light" ? "dark" : "light";
    localStorage.setItem(STORAGE_KEY, next);
    apply(next);
  });
}
