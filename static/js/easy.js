function connect() {
    if ("WebSocket" in window) {
        var ws = new WebSocket("ws://127.0.0.1:8080/easy");
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


var base_x = 1200;
var base_y = 1200;
var multiplier = 265;
var radius = 50;
var svg;
var circle;
var prevDate;
// load the external svg from a file
d3.xml("assets/iBeacon_Layout_Enhanced.svg", "image/svg+xml", function(xml) {
    var importedNode = document.importNode(xml.documentElement, true);
    d3.select("div#vis")
      .each(function() {
        this.appendChild(importedNode);
    });
    svg = d3.select("svg");
});

/*
--------- Replace this method with actual rendering logic ----------
 */
function render(evt) {
    const received_msg = JSON.parse(JSON.parse(evt.data));
    console.log(received_msg[0]);
    var loc = received_msg[0].location;
    var delta_x = loc.charCodeAt(0) - 65;
    var delta_y = parseInt(loc.substring(1))-1;

    if(circle) {
        circle.transition().attr("transform", "translate(" + (base_x+delta_x*multiplier) + "," + (base_y+delta_y*multiplier) + ")").duration(2000);
    } else if(svg) {
        circle = svg.append("circle").attr("cx", 0).attr("cy", 0).attr("r", radius).style("fill", "DeepPink");
        circle.transition().attr("transform", "translate(" + (base_x+delta_x*multiplier) + "," + (base_y+delta_y*multiplier) + ")").duration(0);        
    }
    prevDate = received_msg[0].date;
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
