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
    center: {lat: 39.9927, lng: -0.068},
    zoom: 18
});

var personIcon = {
    url: "assets/person_icon.png", // url
    scaledSize: new google.maps.Size(40, 40), // scaled size
    // origin: new google.maps.Point(0,0), // origin
    // anchor: new google.maps.Point(0, 0) // anchor
};

/*
--------- Replace this method with actual rendering logic ----------
 */
//
var userInfoDict = {};
var markers = {};
function render(evt) {
    const received_messages = JSON.parse(JSON.parse(evt.data));
    for(key in received_messages) {
        received_msg = received_messages[key];
        // if(received_msg.FLOOR == 0) {
        // var latLng = L.utm({x: (received_msg.LONGITUDE/1.3+243888), y: (received_msg.LATITUDE/1.3+689237), zone: 31, southHemi: false}).latLng();
        var latLng = L.utm({x: (received_msg.LONGITUDE/1.275+244004), y: (received_msg.LATITUDE/1.275+615862), zone: 31, southHemi: false}).latLng();
        userInfoDict[received_msg.USERID] = {data: received_msg, latLng: latLng};
        // }
    }
    console.log(userInfoDict);
    for(userID in userInfoDict) {
        var userInfo = userInfoDict[userID];
        if(userID in markers) {
            markers[userID].setPosition({lat: userInfo.latLng.lat, lng: userInfo.latLng.lng})
        } else {
            var marker = new SlidingMarker({
                position: {lat: userInfo.latLng.lat, lng: userInfo.latLng.lng},
                map: map,
                title: userID.toString(),
                icon: personIcon,
                duration: 2000,
                easing: "easeOutExpo"
            });
            markers[userID] = marker;
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


$('#lookup').on('click', function(event) {
    event.preventDefault();
    var userID = $('#userID').val();
    if(userID in userInfoDict) {
        $('#error-text').hide();
        $('#lookup-result').html("<pre><code>" + JSON.stringify(userInfoDict[userID].data, null, 4) + "</pre></code>");
        markers[userID].setAnimation(google.maps.Animation.BOUNCE);
        setTimeout(function(){ markers[userID].setAnimation(null); }, 1500);
    } else {
        $('#error-text').show();
        $('#lookup-result').html("");
    }
  });

