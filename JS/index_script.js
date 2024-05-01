document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("dataset-touch").addEventListener("change", function () {
        if (this.checked) {
            document.getElementById("model-touch").checked = false;
        }
    });
    document.getElementById("model-touch").addEventListener("change", function () {
        if (this.checked) {
            document.getElementById("dataset-touch").checked = false;
        }
    });
});

function openNav() {
    document.getElementById("mySidenav").style.width = "350px";
    document.getElementById("main").style.cssText = "opacity: 0.85; transition: opacity 0.5s ease";
    document.body.style.cssText = "background: repeating-conic-gradient(from 30deg, rgba(0, 0, 0, 0) 0 120deg, rgba(60, 60, 60, 0.85) 0 180deg) calc(0.5 * 150px) calc(0.5 * 150px * 0.577), repeating-conic-gradient(from 30deg, rgba(29, 29, 29, 0.85) 0 60deg, rgba(79, 79, 79, 0.85) 0 120deg, rgba(60, 60, 60, 0.85) 0 180deg); background-size: 150px calc(150px * 0.577);";
}

function closeNav() {
    document.getElementById("dataset-touch").checked = false;
    document.getElementById("model-touch").checked = false;
    document.getElementById("mySidenav").style.width = "0";
    document.getElementById("main").style.opacity = "1";
    document.getElementById("main").style.transition = "opacity 0.5s ease";
    document.body.style.cssText = "background: repeating-conic-gradient(from 30deg, rgba(0, 0, 0, 0) 0 120deg, rgba(60, 60, 60, 1) 0 180deg) calc(0.5 * 150px) calc(0.5 * 150px * 0.577), repeating-conic-gradient(from 30deg, rgba(29, 29, 29, 1) 0 60deg, rgba(79, 79, 79, 1) 0 120deg, rgba(60, 60, 60, 1) 0 180deg); background-size: 150px calc(150px * 0.577);";
}