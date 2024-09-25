document.addEventListener("DOMContentLoaded", function () {
    const userInput = document.getElementById("user-input");
    const sendButton = document.getElementById("send-button");
    const resultContainer = document.querySelector(".result-container h2");

    async function validateAndSearch(event) {
        event.preventDefault(); // Prevent form submission if using a form

        const query = userInput.value.trim();

        if (!query) {
            resultContainer.innerHTML = `<span style="color: red;">Input cannot be empty</span>`;
            return;
        }

        try {
            const response = await fetch("/recommend", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ input: query })
            });

            if (!response.ok) {
                resultContainer.innerHTML = `<span style="color: red;">Error: ${response.statusText}</span>`;
                return;
            }

            const result = await response.json();
            if (result.error) {
                resultContainer.innerHTML = `<span style="color: red;">Error: ${result.error}</span>`;
            } else {
                resultContainer.innerHTML = `<span style="color: blue;">Top 5 Recommended Companies for ${query}:</span><br>`;
                result.forEach(company => {
                    resultContainer.innerHTML += `<p><strong>${company['Company Name']}</strong> - Score: ${company['Total Score \n(out of 100)']} (Rank: ${company['Total Rank']})</p>`;
                });
            }
        } catch (error) {
            resultContainer.innerHTML = `<span style="color: red;">Error: ${error.message}</span>`;
        }
    }

    // Add event listeners
    sendButton.addEventListener("click", validateAndSearch);
    userInput.addEventListener("keyup", function (event) {
        if (event.key === "Enter") {
            validateAndSearch(event);
        }
    });
});