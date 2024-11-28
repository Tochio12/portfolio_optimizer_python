document.addEventListener("DOMContentLoaded", function () {
    const tickerInput = document.getElementById("tickers");
    const suggestionsBox = document.createElement("div");
    suggestionsBox.className = "autocomplete-box";
    tickerInput.parentNode.appendChild(suggestionsBox);

    const stockTickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "DIS", "JPM"];

    tickerInput.addEventListener("input", function () {
        const query = tickerInput.value.split(",").map(t => t.trim().toUpperCase());
        const currentInput = query[query.length - 1]; // Get the last part after the last comma

        suggestionsBox.innerHTML = ""; // Clear old suggestions

        if (currentInput) {
            const filteredSuggestions = stockTickers.filter(ticker => ticker.startsWith(currentInput) && !query.includes(ticker));
            
            filteredSuggestions.forEach(suggestion => {
                const suggestionItem = document.createElement("div");
                suggestionItem.className = "suggestion-item";
                suggestionItem.textContent = suggestion;

                // When suggestion is clicked
                suggestionItem.addEventListener("click", function () {
                    query[query.length - 1] = suggestion; // Replace last input with the suggestion
                    tickerInput.value = query.join(", ") + ", "; // Add a comma for new input
                    suggestionsBox.innerHTML = ""; // Clear suggestions
                    tickerInput.focus(); // Keep the input field active
                });

                suggestionsBox.appendChild(suggestionItem);
            });
        }
    });

    // Hide suggestions when clicking outside the input or suggestions
    document.addEventListener("click", function (event) {
        if (!suggestionsBox.contains(event.target) && event.target !== tickerInput) {
            suggestionsBox.innerHTML = "";
        }
    });
});

