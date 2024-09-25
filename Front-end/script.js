document.addEventListener("DOMContentLoaded", function () {
    const userInput = document.getElementById("user-input");
    const sendButton = document.getElementById("send-button");
    const resultContainer = document.querySelector(".result-container h2");

    // Function to validate and simulate a search
    function validateInput() {
        const query = userInput.value.trim(); // Get the user input

        // Check if input is empty or consists only of spaces
        if (query.length === 0) {
            resultContainer.innerHTML = `<span style="color: red;">Input cannot be empty</span>`;
            return false;
        }
        
        // Check if input contains only letters
        const onlyLetters = /^[a-zA-Z]+$/.test(query);
        if (!onlyLetters) {
            resultContainer.innerHTML = `<span style="color: red;">Input must contain only letters (no numbers or special characters).</span>`;
            return false;
        }

        // Check if input is less than 2 characters
        if (query.length <= 2) {
            resultContainer.innerHTML = `<span style="color: red;">Input must be at least 3 characters long.</span>`;
            return false;
        }

        // If input passes validation
        if (query) {
        resultContainer.innerHTML = `<span style="color: blue;">Choice for: ${query}</span><br><span style="color: blue;">Top 3 Company</span>`;
        return true;
        }
        
    }

    // Event listener for the search button
    sendButton.addEventListener("click", function () {
        validateInput();
    });

    // Optionally, allow pressing 'Enter' to trigger the search
    userInput.addEventListener("keyup", function (event) {
        if (event.key === "Enter") {
            validateInput();
        }
    });
});