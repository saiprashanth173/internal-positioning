function connect() {
    if ("WebSocket" in window) {

        var ws = new WebSocket("ws://127.0.0.1:8080/websocket");


        ws.onmessage = render;
        ws.onclose = function () {
            // websocket is closed.
            alert("Connection is closed...");
        };
    }
    else {
        // The browser doesn't support WebSocket
        alert("WebSocket NOT supported by your Browser!");
    }
}

/*
--------- Replace this method with actual rendering logic ----------
 */
function render(evt) {
    var received_msg = JSON.parse(evt.data);
    for (var i = 0; i < received_msg.length; i++) {
        var x = received_msg[0];
        var element = document.getElementById(x.id);
        let innerHTML = "<b>" + JSON.stringify(x) + "</b>";
        if (element == undefined) {
            document.getElementById('content').innerHTML += (`<div id="${x.id}">${innerHTML} </div>`);
        } else {
            element.innerHTML = innerHTML;
        }
    }
}

window.onload = function () {
    connect();
};
