const conversation = document.querySelector("#conversation");
const welcome = document.querySelector("#welcome");
const form = document.querySelector("#chat-form");
const questionInput = document.querySelector("#question-input");
const imageInput = document.querySelector("#image-input");
const imagePreview = document.querySelector("#image-preview");
const previewImage = document.querySelector("#preview-image");
const previewName = document.querySelector("#preview-name");
const removeImageButton = document.querySelector("#remove-image");
const sendButton = document.querySelector("#send-button");
const backendStatus = document.querySelector("#backend-status");
const statusDot = document.querySelector("#status-dot");

let selectedImage = null;
let previewUrl = null;
const messageImageUrls = new Set();

function scrollToLatest() {
  conversation.scrollTop = conversation.scrollHeight;
}

function addMessage(role, text, imageUrl = null) {
  if (welcome) welcome.hidden = true;

  let list = conversation.querySelector(".message-list");
  if (!list) {
    list = document.createElement("div");
    list.className = "message-list";
    conversation.append(list);
  }

  const message = document.createElement("article");
  message.className = `message ${role}`;
  const label = document.createElement("span");
  label.className = "message-label";
  label.textContent = role === "user" ? "You" : "Carefully";
  message.append(label);

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  if (imageUrl) {
    const image = document.createElement("img");
    image.className = "message-image";
    image.src = imageUrl;
    image.alt = "Image shared for medical explanation";
    bubble.append(image);
  }
  if (text) {
    const textNode = document.createElement("div");
    textNode.textContent = text;
    bubble.append(textNode);
  }
  message.append(bubble);
  list.append(message);
  scrollToLatest();
  return message;
}

function setBackendStatus(state, text) {
  backendStatus.textContent = text;
  statusDot.className = `status-dot ${state}`;
}

async function checkBackend() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error(`Health check failed (${response.status}).`);
    const data = await response.json();
    if (data.rag_ready) {
      setBackendStatus("ready", "Assistant ready");
    } else {
      setBackendStatus("", "Assistant is starting");
    }
  } catch (error) {
    setBackendStatus("offline", "Assistant unavailable");
  }
}

function clearSelectedImage() {
  selectedImage = null;
  imageInput.value = "";
  imagePreview.hidden = true;
  previewImage.removeAttribute("src");
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
}

imageInput.addEventListener("change", () => {
  const image = imageInput.files[0];
  if (!image) return;
  if (!["image/png", "image/jpeg"].includes(image.type)) {
    addMessage("assistant", "Please choose a PNG or JPEG image.");
    clearSelectedImage();
    return;
  }
  if (image.size > 5 * 1024 * 1024) {
    addMessage("assistant", "That image is too large. Please choose an image under 5 MB.");
    clearSelectedImage();
    return;
  }
  selectedImage = image;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(image);
  previewImage.src = previewUrl;
  previewName.textContent = image.name;
  imagePreview.hidden = false;
});

removeImageButton.addEventListener("click", clearSelectedImage);

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => {
    questionInput.value = button.dataset.question;
    questionInput.focus();
    questionInput.dispatchEvent(new Event("input"));
  });
});

document.querySelector("#new-chat").addEventListener("click", () => {
  const messages = conversation.querySelector(".message-list");
  if (messages) messages.remove();
  messageImageUrls.forEach((url) => URL.revokeObjectURL(url));
  messageImageUrls.clear();
  if (welcome) welcome.hidden = false;
  clearSelectedImage();
  questionInput.value = "";
  questionInput.focus();
});

questionInput.addEventListener("input", () => {
  questionInput.style.height = "24px";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 130)}px`;
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  const image = selectedImage;
  if (!question && !image) {
    questionInput.focus();
    return;
  }
  if (!image && question.length < 2) {
    addMessage("assistant", "Please enter a question with at least two characters.");
    return;
  }

  const imageUrl = image ? URL.createObjectURL(image) : null;
  if (imageUrl) messageImageUrls.add(imageUrl);
  addMessage("user", question || "Please explain this medical image.", imageUrl);
  questionInput.value = "";
  questionInput.style.height = "24px";
  clearSelectedImage();
  sendButton.disabled = true;

  const pending = addMessage("assistant", "");
  const pendingBubble = pending.querySelector(".message-bubble");
  const dots = document.createElement("span");
  dots.className = "typing-indicator";
  dots.setAttribute("aria-label", "Generating answer");
  for (let index = 0; index < 3; index += 1) dots.append(document.createElement("span"));
  pendingBubble.append(dots);

  try {
    let response;
    if (image) {
      const body = new FormData();
      body.append("image", image);
      body.append("question", question);
      response = await fetch("/analyze-image", { method: "POST", body });
    } else {
      response = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
    }
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "The request could not be completed.");
    pendingBubble.replaceChildren();
    pendingBubble.textContent = data.answer;
    scrollToLatest();
  } catch (error) {
    pendingBubble.classList.add("error-message");
    pendingBubble.textContent = error.message || "Unable to reach the assistant. Please try again.";
  } finally {
    sendButton.disabled = false;
    questionInput.focus();
  }
});

window.addEventListener("pagehide", () => {
  messageImageUrls.forEach((url) => URL.revokeObjectURL(url));
  if (previewUrl) URL.revokeObjectURL(previewUrl);
});

checkBackend();
window.setInterval(checkBackend, 30000);
