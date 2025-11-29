// js/main.js

document.addEventListener('DOMContentLoaded', () => {
    const body = document.body;

    // Header
    const themeToggleBtn = document.getElementById('themeToggleBtn');
    const fontToggleBtn = document.getElementById('fontToggleBtn');

    // InitialScreen
    const sendBtn = document.getElementById('sendBtn');
    const textarea = document.getElementById('email_body');
    const initialScreen = document.getElementById('screen-initial');

    // ResultScreen
    const resultScreen = document.getElementById('screen-result');
    const resultStatusEl = document.getElementById('resultStatus');
    const resultReasonEl = document.getElementById('resultReason');
    const resultPromptEl = document.getElementById('resultPrompt');
    const nextBtn = document.getElementById('nextBtn');

    // Loading indicator
    const loadingIndicator = document.getElementById('loadingIndicator');

    let isDarkMode = false;
    const fontClasses = ['font-small', 'font-medium', 'font-large'];
    let currentFontIndex = 1;
    let isLoading = false;

    // Theme/Font
    function applyTheme() {
        if (isDarkMode) {
            body.classList.add('theme-dark');
            body.classList.remove('theme-light');
            themeToggleBtn?.setAttribute('aria-pressed', 'true');
        } else {
            body.classList.add('theme-light');
            body.classList.remove('theme-dark');
            themeToggleBtn?.setAttribute('aria-pressed', 'false');
        }
    }

    function applyFontSize() {
        fontClasses.forEach(c => body.classList.remove(c));
        body.classList.add(fontClasses[currentFontIndex]);
    }

    // Send Button
    function updateSendButtonState() {
        if (!sendBtn || !textarea) return;
        const hasText = textarea.value.trim().length > 0;
        sendBtn.disabled = isLoading || !hasText;
    }

    // Render Initial Screen
    function showInitialScreen() {
        if (!initialScreen || !resultScreen || !textarea) return;
        initialScreen.classList.remove('hidden');
        resultScreen.classList.add('hidden');

        textarea.value = '';
        updateSendButtonState();
        textarea.focus();
    }

    // Loading ON/OFF helper
    function setLoading(loading) {
        isLoading = loading;

        if (loadingIndicator) {
            loadingIndicator.classList.toggle('hidden', !loading);
        }
        if (textarea) {
            textarea.disabled = loading;
        }
        updateSendButtonState();
    }


    /**
     * ResultScreen Rendering
     * @param {{ isSpam: boolean, reason: string }} result - Result object from Backend
     * @param {string} userPrompt - Prompt contents that user writes
     */
    function showResultScreen(result, userPrompt) {
        if (!initialScreen || !resultScreen || !resultStatusEl || !resultReasonEl || !resultPromptEl) return;

        initialScreen.classList.add('hidden');
        resultScreen.classList.remove('hidden');

        const isSpam = !!result.isSpam;

        resultStatusEl.textContent = isSpam ? 'SPAM' : 'Not Spam';
        resultStatusEl.classList.toggle('result-status-spam', isSpam);
        resultStatusEl.classList.toggle('result-status-notspam', !isSpam);

        resultReasonEl.textContent = result.reason || '';

        resultPromptEl.textContent = userPrompt || '';
    }

    // InitialScreen의 handleSubmit
    function handleSubmit() {
        if (!textarea) return;

        const value = textarea.value.trim();
        if (!value) return;

        setLoading(true);

        // ================================
        //    BACKEND INTEGRATION POINT
        // ================================
        //
        // What Backend should do:
        // 1) `value` (Email text that user wrote) should be passed
        //    to ML Spam detector API,
        // 2) Response(JSON) in the correct form and put into
        //    showResultScreen(result, value)and call it.
        //
        // Example of expected result:
        //
        // const result = {
        //   isSpam: true,            // or false
        //   reason: "The reason why..." // Can be multiple sentences
        // };
        //
        // Async example (fetch):
        //
        // fetch('/api/spam-check', {
        //   method: 'POST',
        //   headers: { 'Content-Type': 'application/json' },
        //   body: JSON.stringify({ content: value })
        // })
        //   .then(res => res.json())
        //   .then(data => {
        //       showResultScreen(data, value);
        //   })
        //   .catch(err => {
        //       console.error(err);
        //       // Error handling UI
        //   });
        //
        // Currently using dummy items for testing in frontend process.
        setTimeout(() => {
            const dummyResult = {
                isSpam: true,
                reason:
                    'This is a dummy explanation from the frontend.\n\n' +
                    'Backend team:\n' +
                    '- Replace handleSubmit() logic in js/main.js\n' +
                    '- Send `value` to your ML model\n' +
                    '- Call showResultScreen({ isSpam, reason }, value) with real data.'
            };

            showResultScreen(dummyResult, value);

            setLoading(false);
        }, 1200);
    }

    // Event Handler

    // DarkMode Toggle
    themeToggleBtn?.addEventListener('click', () => {
        isDarkMode = !isDarkMode;
        applyTheme();
    });

    // Font size
    fontToggleBtn?.addEventListener('click', () => {
        currentFontIndex = (currentFontIndex + 1) % fontClasses.length;
        applyFontSize();
    });

    // Send button and Textarea
    if (sendBtn && textarea) {
        sendBtn.addEventListener('click', (e) => {
            e.preventDefault();
            handleSubmit();
        });

        textarea.addEventListener('input', () => {
            updateSendButtonState();
        });

        // Enter for send, Shift+Enter for line break
        textarea.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSubmit();
            }
        });
    }

    // Next button -> Initial Screen render
    nextBtn?.addEventListener('click', (e) => {
        e.preventDefault();
        showInitialScreen();
    });

    // Initial apply
    applyTheme();
    applyFontSize();
    updateSendButtonState();
});
