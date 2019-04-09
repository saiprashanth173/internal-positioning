function connect() {
    if ("WebSocket" in window) {
        var ws = new WebSocket("ws://127.0.0.1:8080/multibuilding");
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

var map = new google.maps.Map(document.getElementById('map'), {
    center: {lat: 39.993, lng: -0.068},
    zoom: 18
});

  // {% for i in range(result|length) %}
  // var marker = new google.maps.Marker({
  //   position: { lat: {{result[i][names_dict["LATITUDE"]]}}, lng: {{result[i][names_dict["LONGITUDE"]]}} },
  //   label: "{{i+1}}",
  //   map: map
  // });
  // {% endfor %}

/*
--------- Replace this method with actual rendering logic ----------
 */
// , 267070, 398375
function render(evt) {
    const received_messages = JSON.parse(JSON.parse(evt.data));
    console.log(received_messages);
    for(key in received_messages) {
        received_msg = received_messages[key];
        var latLng = L.utm({x: (received_msg.LONGITUDE/1.3+243888), y: (received_msg.LATITUDE/1.3+689237), zone: 31, southHemi: false}).latLng();
        var latLng = L.utm({x: (received_msg.LONGITUDE/1.275+244004), y: (received_msg.LATITUDE/1.275+615862), zone: 31, southHemi: false}).latLng();
        // console.log(
            // latLng
        // )
        if(received_msg.FLOOR == 0) {
            var marker = new google.maps.Marker({
                position: {lat: latLng.lat, lng: latLng.lng},
                map: map,
                title: received_msg.BUILDINGID.toString()
            });
        }

    }
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
