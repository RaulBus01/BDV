fetch("https://api.sofascore.com/api/v1/event/14083261/incidents", {
  "referrer": "https://www.sofascore.ro/",
  "body": null,
  "method": "GET",
  "mode": "cors",
  "credentials": "omit"
}).then(response => response.json()).then(data => console.log(data)).catch(error => console.error('Error:', error));

