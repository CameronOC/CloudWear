/*
	Dopetrope by HTML5 UP
	html5up.net | @ajlkn
	Free for personal and commercial use under the CCA 3.0 license (html5up.net/license)
*/

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
    
var urlArray = ['images/shirt.jpg','images/pants.jpg','images/hat.jpg'];
    var count = -1;

      function nextUrl() {
        url = urlArray[++count];
        count = (count >= urlArray.length - 1) ? -1 : count;
        return url;
      }

var HEADlinkforward = nextUrl()
var HEADlinkback = nextUrl()

var TOPlinkforward = nextUrl()
var TOPlinkback = nextUrl()

var BOTTOMlinkforward = nextUrl()
var BOTTOMlinkback = nextUrl()

    $(function() {
 $('.button-1-left').click(function(e){
     e.preventDefault();
   $("#head").attr('src',HEADlinkback);
 });
});
    
    $(function() {
 $('.button-1-right').click(function(e){
     e.preventDefault();
   $("#head").attr('src',HEADlinkforward);
 });
});
    
    $(function() {
 $('.button-2-left').click(function(e){
     e.preventDefault();
   $("#top").attr('src',TOPlinkback);
 });
});
    
    $(function() {
 $('.button-2-right').click(function(e){
     e.preventDefault();
   $("#top").attr('src',TOPlinkforward);
 });
});
    
    $(function() {
 $('.button-3-left').click(function(e){
     e.preventDefault();
   $("#bottom").attr('src',BOTTOMlinkback);
 });
});
    
    $(function() {
 $('.button-3-right').click(function(e){
     e.preventDefault();
   $("#bottom").attr('src',BOTTOMlinkforward);
 });
});

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