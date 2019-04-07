function connect() {
    if ("WebSocket" in window) {

        var ws = new WebSocket("ws://127.0.0.1:8080/websocket");


        ws.onmessage = render;
        ws.onclose = function () {
            // websocket is closed.
            // alert("Connection is closed...");
        };
    }
    else {
        // The browser doesn't support WebSocket
        alert("WebSocket NOT supported by your Browser!");
    }
}


// load the external svg from a file
d3.xml("assets/iBeacon_Layout_Enhanced.svg", "image/svg+xml", function(xml) {
    var importedNode = document.importNode(xml.documentElement, true);
    d3.select("div#vis")
      .each(function() {
        this.appendChild(importedNode);
      });
});

var svg = d3.select("svg");
var circle = svg.append("circle").attr("cx", 30).attr("cy", 30).attr("r", 20);


/*
--------- Replace this method with actual rendering logic ----------
 */
function render(evt) {
    const received_msg = JSON.parse(JSON.parse(evt.data));
    console.log(received_msg);
    // for (let i = 0; i < received_msg.length; i++) {
        // const x = received_msg[i];
        // const element = document.getElementById("out");
        // const innerHTML = "<b>" + JSON.stringify(x) + "</b>";
        // if (element == undefined) {
        // document.getElementById('content').innerHTML = (`<div id="${x.beacon_id}">${innerHTML} </div>`);
        // } else {
        //     element.innerHTML = innerHTML;
        // }
    // }
}

window.onload = function () {
    connect();
};
