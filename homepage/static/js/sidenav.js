function openNav() {
  document.getElementById("mobnav").style.height = "100%";
}

/* Set the width of the sidebar to 0 and the left margin of the page content to 0 */
function closeNav() {
  document.getElementById("mobnav").style.height = "0";
}

function showCarregando() {
  var element = document.getElementById("carregando");
  element.classList.add("lds-dual-ring");
}
document.getElementById("tvalimg").style.width= (screen.width*0.6).toString()+'px';
