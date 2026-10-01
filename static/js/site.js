// Small page behaviours: play clips only while visible, the demo iframe, the video dialog, BibTeX copy.

(function () {
  // Clips load and play only when on screen, so the page stays light.
  var clips = document.querySelectorAll('video[data-autoplay]');
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        var v = e.target;
        if (e.isIntersecting) {
          if (v.preload === 'none') { v.preload = 'auto'; v.load(); }
          var p = v.play();
          if (p && p.catch) p.catch(function () {});
        } else {
          v.pause();
        }
      });
    }, { threshold: 0.25 });
    clips.forEach(function (v) { v.removeAttribute('autoplay'); io.observe(v); });
  }

  // Side-by-side pairs start together; the shorter clip holds its last frame
  // until both have finished, then both restart.
  var pairs = document.querySelectorAll('[data-pair]');
  pairs.forEach(function (pair) {
    var vids = Array.prototype.slice.call(pair.querySelectorAll('video'));
    var done = 0;
    var visible = false;
    function playAll() {
      vids.forEach(function (v) {
        if (v.ended) return;          // finished first: wait for its partner
        if (v.preload === 'none') { v.preload = 'auto'; v.load(); }
        var p = v.play();
        if (p && p.catch) p.catch(function () {});
      });
    }
    vids.forEach(function (v) {
      v.addEventListener('ended', function () {
        done += 1;
        if (done < vids.length) return;
        done = 0;
        vids.forEach(function (w) { w.currentTime = 0; });
        if (visible) playAll();
      });
    });
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (entries) {
        visible = entries[0].isIntersecting;
        if (visible) playAll(); else vids.forEach(function (v) { v.pause(); });
      }, { threshold: 0.25 }).observe(pair);
    } else {
      visible = true;
      playAll();
    }
  });

  // The demo iframe is created on click, so visitors who never open it download none of it.
  var launch = document.getElementById('demo-launch');
  if (launch) {
    launch.addEventListener('click', function () {
      var frame = document.getElementById('demo-frame');
      var iframe = document.createElement('iframe');
      iframe.title = 'CSF interactive demo';
      iframe.src = 'demo/';
      iframe.allow = 'fullscreen';
      frame.appendChild(iframe);
      launch.remove();
    });
  }

  var dialog = document.getElementById('video-dialog');
  var full = document.getElementById('full-video');
  document.getElementById('open-video').addEventListener('click', function () {
    dialog.showModal();
    var p = full.play();
    if (p && p.catch) p.catch(function () {});
  });
  function closeVideo() { full.pause(); dialog.close(); }
  document.getElementById('close-video').addEventListener('click', closeVideo);
  dialog.addEventListener('click', function (e) { if (e.target === dialog) closeVideo(); });
  dialog.addEventListener('cancel', function () { full.pause(); });

  var copy = document.getElementById('copy-bibtex');
  copy.addEventListener('click', function () {
    var text = document.getElementById('bibtex-code').innerText;
    navigator.clipboard.writeText(text).then(function () {
      copy.textContent = 'Copied';
      setTimeout(function () { copy.textContent = 'Copy'; }, 1500);
    });
  });
})();
