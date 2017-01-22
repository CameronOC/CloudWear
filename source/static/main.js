/*
	Dopetrope by HTML5 UP
	html5up.net | @ajlkn
	Free for personal and commercial use under the CCA 3.0 license (html5up.net/license)
*/
var count_top=0;
var count_middle=0;
var count_bottom=0;

(function($) {

	skel
		.breakpoints({
			desktop: '(min-width: 737px)',
			tablet: '(min-width: 737px) and (max-width: 1200px)',
			mobile: '(max-width: 736px)'
		})
		.viewport({
			breakpoints: {
				tablet: {
					width: 1080
				}
			}
		});

var TopURLArray = ["{{url_for('static', filename='trucker.jpg')}}","{{url_for('static', filename='fedora.jpg')}}","{{url_for('static', filename='hat.jpg')}}"];
var MiddleURLArray = ["{{url_for('static', filename='shirt.jpg')}}","{{url_for('static', filename='blueshirt.jpg')}}","{{url_for('static', filename='acid.png')}}"];
var BottomURLArray = ["{{url_for('static', filename='fleecepants.jpg')}}","{{url_for('static', filename='pants.jpg')}}","{{url_for('static', filename='navypants.jpg')}}"];

function TopnextUrl() {
        window.count_top++;
        window.count_top = (window.count_top >= TopURLArray.length) ? 0 : window.count_top;
        url = TopURLArray[window.count_top];
        return url;
      }

function TopbackUrl() {
        window.count_top--;
        window.count_top = (window.count_top < 0) ? TopURLArray.length : window.count_top;
        url = TopURLArray[window.count_top];
        return url;
      }

function MiddlenextUrl() {
        window.count_middle++;
        window.count_middle = (window.count_middle >= MiddleURLArray.length) ? 0 : window.count_middle;
        url = MiddleURLArray[window.count_middle];
        return url;
      }

function MiddlebackUrl() {
        window.count_middle--;
        window.count_middle = (window.count_middle < 0) ? MiddleURLArray.length : window.count_middle;
        url = MiddleURLArray[window.count_middle];
        return url;
      }

function BottomnextUrl() {
        window.count_bottom++;
        window.count_bottom = (window.count_bottom >= BottomURLArray.length) ? 0 : window.count_bottom;
        url = BottomURLArray[window.count_bottom];
        return url;
      }

function BottombackUrl() {
        window.count_bottom--;
        window.count_bottom = (window.count_bottom < 0) ? BottomURLArray.length : window.count_bottom;
        url = BottomURLArray[window.count_bottom];
        return url;
      }

    $(function() {
 $('.button-1-left').click(function(e){
     e.preventDefault();
   $("#top").attr('src',TopbackUrl());
 });
});

    $(function() {
 $('.button-1-right').click(function(e){
     e.preventDefault();
   $("#top").attr('src',TopnextUrl());
 });
});

    $(function() {
 $('.button-2-left').click(function(e){
     e.preventDefault();
   $("#middle").attr('src',MiddlebackUrl());
 });
});

    $(function() {
 $('.button-2-right').click(function(e){
     e.preventDefault();
   $("#middle").attr('src',MiddlenextUrl());
 });
});

    $(function() {
 $('.button-3-left').click(function(e){
     e.preventDefault();
   $("#bottom").attr('src',BottombackUrl());
 });
});

    $(function() {
 $('.button-3-right').click(function(e){
     e.preventDefault();
   $("#bottom").attr('src',BottomnextUrl());
 });
});
<video autoplay></video>
<img src="">
<canvas style="display:none;"></canvas>

<script>
 var video = document.querySelector('video');
 var canvas = document.querySelector('canvas');
 var ctx = canvas.getContext('2d');
 var localMediaStream = null;

 function snapshot() {
   if (localMediaStream) {
     ctx.drawImage(video, 0, 0);
     // "image/webp" works in Chrome.
     // Other browsers will fall back to image/png.
     document.querySelector('img').src = canvas.toDataURL('image/webp');
   }
 }

 video.addEventListener('click', snapshot, false);

 // Not showing vendor prefixes or code that works cross-browser.
 navigator.getUserMedia({video: true}, function(stream) {
   video.src = window.URL.createObjectURL(stream);
   localMediaStream = stream;
 }, errorCallback);
</script>


<div style="text-align:center;">
 <video id="screenshot-stream" class="videostream" autoplay></video>
 <img id="screenshot" src="">
 <canvas id="screenshot-canvas" style="display:none;"></canvas>
 <p><button id="screenshot-button">Capture</button> <button id="screenshot-stop-button">Stop</button></p>
</div>
	$(function() {

		var	$window = $(window),
			$body = $('body');

		// Disable animations/transitions until the page has loaded.
			$body.addClass('is-loading');

			$window.on('load', function() {
				$body.removeClass('is-loading');
			});

		// Fix: Placeholder polyfill.
			$('form').placeholder();

		// Prioritize "important" elements on mobile.
			skel.on('+mobile -mobile', function() {
				$.prioritize(
					'.important\\28 mobile\\29',
					skel.breakpoint('mobile').active
				);
			});

		// Dropdowns.
			$('#nav > ul').dropotron({
				mode: 'fade',
				noOpenerFade: true,
				alignment: 'center'
			});

		// Off-Canvas Navigation.

			// Title Bar.
				$(
					'<div id="titleBar">' +
						'<a href="#navPanel" class="toggle"></a>' +
					'</div>'
				)
					.appendTo($body);

			// Navigation Panel.
				$(
					'<div id="navPanel">' +
						'<nav>' +
							$('#nav').navList() +
						'</nav>' +
					'</div>'
				)
					.appendTo($body)
					.panel({
						delay: 500,
						hideOnClick: true,
						hideOnSwipe: true,
						resetScroll: true,
						resetForms: true,
						side: 'left',
						target: $body,
						visibleClass: 'navPanel-visible'
					});

			// Fix: Remove navPanel transitions on WP<10 (poor/buggy performance).
				if (skel.vars.os == 'wp' && skel.vars.osVersion < 10)
					$('#titleBar, #navPanel, #page-wrapper')
						.css('transition', 'none');

	});

})(jQuery);

// Grab elements, create settings, etc.
var video = document.getElementById('video');

// Get access to the camera!
if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
   // Not adding `{ audio: true }` since we only want video now
   navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
       video.src = window.URL.createObjectURL(stream);
       video.play();
   });
}

// Elements for taking the snapshot
var canvas = document.getElementById('canvas');
var context = canvas.getContext('2d');
var video = document.getElementById('video');

// Trigger photo take
document.getElementById("snap").addEventListener("click", function() {
    context.drawImage(video, 0, 0, 640, 480);
});