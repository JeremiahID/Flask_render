
document.getElementById("form").addEventListener("submit",
function(event) {
    event.preventDefault();
    const email = document.getElementById("email").value;
    const password = document.getElementById("password").value;
    const data ={ email: email, pasword: password};
    fetch("/process", {
        method: "POST" ,
        headers:{
            "Content-Type": "application/json"
        },
        body:JSON.stringify(data)
    })
    .then(Response => Response.json())
    .then(data => console.log(data));
});