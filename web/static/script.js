document.addEventListener('DOMContentLoaded', function() {
          var dropArea = document.getElementById('drop-area');

          // Prevent default drag behaviors
          ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
              dropArea.addEventListener(eventName, preventDefaults, false);
              document.body.addEventListener(eventName, preventDefaults, false);
          });

          // Highlight drop area when item is dragged over it
          ['dragenter', 'dragover'].forEach(eventName => {
              dropArea.addEventListener(eventName, highlight, false);
          });

          // Remove highlight when item is dragged out of drop area
          ['dragleave', 'drop'].forEach(eventName => {
              dropArea.addEventListener(eventName, unhighlight, false);
          });

          // Handle dropped files
          dropArea.addEventListener('drop', handleDrop, false);

          function preventDefaults(event) {
              event.preventDefault();
              event.stopPropagation();
          }

          function highlight() {
              dropArea.classList.add('highlight');
          }

          function unhighlight() {
              dropArea.classList.remove('highlight');
          }

          function handleDrop(event) {
              var dt = event.dataTransfer;
              var files = dt.files;
              handleFiles(files);
          }

          function handleFiles(files) {
              // Handle uploaded files here
              var file = files[0]; // Assuming only one file is dropped
              document.getElementById('file-input').files = files; // Populate file input with dropped file
              document.getElementById('file-label').textContent = file.name; // Update label with file name
          }

          // Add event listener to file input for manual file selection
          document.getElementById('file-input').addEventListener('change', function(event) {
              var file = event.target.files[0];
              document.getElementById('file-label').textContent = file.name;
          });

          // Handle form submission
          document.getElementById('upload-form').addEventListener('submit', function(event) {
              event.preventDefault();
              var form = event.target;
              var formData = new FormData(form);
              fetch(form.action, {
                  method: 'POST',
                  body: formData
              })
              .then(response => response.json())
              .then(data => {
                  // Handle response (if needed)
                  console.log(data);
                  // Optionally, you can update the UI to display the processed image
              })
              .catch(error => console.error('Error:', error));
          });

          // JavaScript functions for image navigation
          var currentIndex = 0;
          var imageNames = [
              {% for image_name in image_names %}
                  "{{ image_name }}",
              {% endfor %}
          ];

          function changeImage(offset) {
              currentIndex += offset;
              if (currentIndex < 0) {
                  currentIndex = imageNames.length - 1;
              } else if (currentIndex >= imageNames.length) {
                  currentIndex = 0;
              }
              var imageElement = document.getElementById("image");
              imageElement.src = "{{ url_for('static', filename='images/') }}" + imageNames[currentIndex];
              imageElement.alt = imageNames[currentIndex];
          }

          document.addEventListener("keydown", function(event) {
              if (event.keyCode === 37) {  // Left arrow key
                  changeImage(-1);
              } else if (event.keyCode === 39) {  // Right arrow key
                  changeImage(1);
              }
          });

          var lastScrollTop = 0;

          window.addEventListener("scroll", function() {
              var currentScroll = window.pageYOffset || document.documentElement.scrollTop;

              if (currentScroll > lastScrollTop) {
                  // Scrolling down
                  var currentSection = document.querySelector('.fade-in');
                  if (currentSection.nextElementSibling) {
                      currentSection.classList.remove('fade-in');
                      currentSection.classList.add('fade-out');
                      currentSection.nextElementSibling.classList.add('fade-in');
                  }
              } else {
                  // Scrolling up
                  var currentSection = document.querySelector('.fade-in');
                  if (currentSection.previousElementSibling) {
                      currentSection.classList.remove('fade-in');
                      currentSection.classList.add('fade-out');
                      currentSection.previousElementSibling.classList.add('fade-in');
                  }
              }

              lastScrollTop = currentScroll <= 0 ? 0 : currentScroll;
          });
      });

      function toggleFavorite() {
          var heartIcon = document.getElementById('heartIcon');
          var imageSrc = document.getElementById('image').src;

          if (heartIcon.classList.contains('bxs-heart')) {
              // Remove from favorites (bxs-heart -> bx-heart)
              heartIcon.classList.remove('bxs-heart');
              heartIcon.classList.add('bx-heart');
              removeFromFavorites(imageSrc);
          } else {
              // Add to favorites (bx-heart -> bxs-heart)
              heartIcon.classList.remove('bx-heart');
              heartIcon.classList.add('bxs-heart');
              addToFavorites(imageSrc);
          }
      }

      function addToFavorites(imageSrc) {
          // Send an AJAX request to add the image to fav.csv
          var xhr = new XMLHttpRequest();
          xhr.open('POST', '/add-to-favorites', true);
          xhr.setRequestHeader('Content-Type', 'application/json');
          xhr.onreadystatechange = function () {
              if (xhr.readyState === 4 && xhr.status === 200) {
                  console.log('Image added to favorites.');
              }
          };
          xhr.send(JSON.stringify({ imageSrc: imageSrc }));
      }

      function removeFromFavorites(imageSrc) {
          // Send an AJAX request to remove the image from fav.csv
          var xhr = new XMLHttpRequest();
          xhr.open('POST', '/remove-from-favorites', true);
          xhr.setRequestHeader('Content-Type', 'application/json');
          xhr.onreadystatechange = function () {
              if (xhr.readyState === 4 && xhr.status === 200) {
                  console.log('Image removed from favorites.');
              }
          };
          xhr.send(JSON.stringify({ imageSrc: imageSrc }));
      }