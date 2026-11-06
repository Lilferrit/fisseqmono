// Esc closes an open modal, through its close button so Dash sees the click.
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  document.querySelectorAll(".sv-modal.open .sv-modal-close").forEach((b) => b.click());
});
