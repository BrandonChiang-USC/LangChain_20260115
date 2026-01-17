$(function () {
    $("#submit").click(chatWithData);
    $("#message").keypress(function (e) {
        if (e.which == 13) {
            chatWithData();
        }
    });
});

function chatWithData() {
    var message = $("#message").val();
    $("#dialog").append("我 : " + message + "\n");
    var data = {
        message: message
    };
    $.post("/call_data", data, function (data) {
        console.log("data:", data);
        let json_data = JSON.parse(data);
        for(let i = 0; i < json_data.length; i++) {
            if(json_data[i][0] == "text") {
                $("#dialog").append("AI : " + json_data[i][1] + "\n");
            }
            else if (json_data[i][0] == "image") {
                $("#dialog").append("AI : " + "霹哩啪拉～～變" + "\n");
                $("#output").attr("src", "/static/images/gemini_output.jpg?" + new Date().getTime());
            }
        }
        $("#dialog").scrollTop($("#dialog")[0].scrollHeight);
    });
    $("#message").val("");
    $("#dialog").scrollTop($("#dialog")[0].scrollHeight);
}