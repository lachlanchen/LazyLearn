(() => {
  const themeToggle = document.getElementById("theme-toggle");
  const themeKey = "lazylearn_theme";

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    if (!themeToggle) return;
    themeToggle.setAttribute("aria-pressed", theme === "dark");
    themeToggle.textContent = theme === "dark" ? "Dark" : "Light";
  }

  let storedTheme = null;
  try {
    storedTheme = localStorage.getItem(themeKey);
  } catch (err) {
    // Ignore storage errors in private or restricted browser modes.
  }

  applyTheme(storedTheme === "dark" ? "dark" : "light");

  if (!themeToggle) return;
  themeToggle.addEventListener("click", () => {
    const nextTheme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(nextTheme);
    try {
      localStorage.setItem(themeKey, nextTheme);
    } catch (err) {
      // Ignore storage errors in private or restricted browser modes.
    }
  });
})();
