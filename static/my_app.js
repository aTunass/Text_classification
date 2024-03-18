let searchText = document.getElementById("search_box").value;
const searchButton = document.getElementById('search_button');
let resultClassification = document.getElementById('result');
//check button
searchButton.addEventListener('click', function() {
    searchText = document.getElementById("search_box").value;
    Clasification_Text(searchText);
});
//function

function Clasification_Text(text) {
    console.log('Classification')
    fetch(`/text_classification?text=${text}`)
    .then(response => response.json())
    .then(data => {
        result =data.results
        resultClassification.textContent = "Result: " + result;
    })
    .catch(error => console.error('Error classification:', error));
}